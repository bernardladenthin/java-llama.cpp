<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# What the CPU backend modules are worth, per microarchitecture

Measured **2026-10-11** against the natives jars CI itself built at llama.cpp **b11538**
(`cpu-linux-x86-64`, `cpu-linux-aarch64`), on rented hardware, **one module staged at a time on the
same machine with the same model**. That staging is the whole point: ggml keeps exactly one module at
run time, so a module is otherwise only ever exercised where the CPU prefers it.

**Method.** Each run gets a directory holding `libjllama`, `ggml`, `ggml-base`, `ggml-rpc` and
**one** `ggml-cpu-<level>` module, with a `jllama-files.txt` naming just that module — so
`ggml_backend_load_best` has exactly one candidate and no choice to make. Model
**Qwen3-0.6B Q4_K_M**, 257-token prompt, 128 tokens generated, `-ngl 0`, threads = vCPU count. `pp`
is prompt processing and `tg` token generation, both in tokens/s as llama.cpp's own timings report
them.

## Intel Granite Rapids — Xeon 6975P-C, 4 vCPU (`m8i.xlarge`)

The first machine on which **all 14 x86 modules load and run**: Granite Rapids carries every feature
set below AMX as well, so this single table is the complete x86 coverage the project had been missing.

| module | pp512 | tg128 | against `haswell` (pp) |
|---|---:|---:|---:|
| **`sapphirerapids`** (AMX) ← ggml's choice | **550.0** | 36.0 | **+148 %** |
| `cascadelake` | 330.8 | 38.8 | +49 % |
| `zen4` | 328.8 | **40.9** | +48 % |
| `icelake` | 328.8 | 37.9 | +48 % |
| `cooperlake` | 328.5 | 40.2 | +48 % |
| `skylakex` | 308.9 | 40.1 | +39 % |
| `cannonlake` | 307.3 | 38.9 | +39 % |
| `haswell` | 221.8 | 40.2 | — |
| `alderlake` | 220.0 | 40.0 | −1 % |
| `piledriver` | 146.2 | 32.9 | −34 % |
| `ivybridge` | 141.2 | 31.0 | −36 % |
| `sandybridge` | 138.1 | 16.9 | −38 % |
| `sse42` | 9.1 | 5.4 | −96 % |
| `x64` | 8.3 | 4.8 | −96 % |

Full set on the same machine: ggml chose `sapphirerapids`, 546.3 / 35.0 — within noise of the staged
run, so the staging does not distort anything.

**AMX is worth +148 % on prompt processing** over the Haswell baseline. That is the number the
"benchmark on AVX-512/VNNI/AMX hardware" item had been asking for, and it is the strongest argument
for shipping all 14 modules rather than one.

**The baseline is far worse than previously assumed.** The project's earlier figure for the plain
`x64` module was "10.4× slower at prompt processing than `haswell`", measured on Zen 3. Here it is
**66× slower than the chosen module** and 27× slower than `haswell`. A baseline-only build would not
be "slow", it would be unusable.

## AMD EPYC Genoa / Zen 4 — EPYC 9R14, 4 vCPU (`m7a.xlarge`)

| module | pp512 | tg128 | against `haswell` (pp) |
|---|---:|---:|---:|
| `cascadelake` | **595.1** | 55.6 | +41 % |
| **`zen4`** ← ggml's choice | 569.3 | 55.2 | +35 % |
| `icelake` | 584.1 | 54.5 | +38 % |
| `cannonlake` | 540.1 | 55.3 | +28 % |
| `skylakex` | 532.0 | 55.3 | +26 % |
| `cooperlake` | 466.5 | 44.6 | +11 % |
| `haswell` | 422.0 | **60.5** | — |
| `piledriver` | 278.3 | 57.5 | −34 % |
| `ivybridge` | 270.2 | 56.2 | −36 % |
| `sandybridge` | 258.5 | 21.8 | −39 % |
| `sse42` | 16.5 | 9.5 | −96 % |
| `x64` | 15.0 | 8.1 | −96 % |
| `sapphirerapids` | — | — | **declines: no AMX** |
| `alderlake` | — | — | **declines: no AVX-VNNI** |

Full set: ggml chose `zen4`, 594.8 / 54.1.

The two declining rows are **correct behaviour, not failures**: the module registers no CPU backend,
the load fails cleanly with `make_cpu_buft_list: no CPU backend found` and a `LlamaException`. Zen 4
has `avx512_vnni` but not the 256-bit `AVX-VNNI` that `alderlake` is built for, and no AMX at all.

## AWS Graviton 4 / Neoverse V2 — 4 vCPU (`c8g.xlarge`)

`sve sve2 i8mm bf16`, no SME.

| module | needs | pp512 | tg128 |
|---|---|---:|---:|
| **`armv8.6_2`** ← ggml's choice | DOTPROD FP16 SVE MATMUL_INT8 SVE2 | 198.0 | 63.6 |
| `armv8.6_1` | DOTPROD FP16 SVE MATMUL_INT8 | 197.7 | 60.8 |
| `armv8.2_3` | DOTPROD FP16 SVE | 176.7 | 58.2 |
| **`armv8.2_2`** | DOTPROD FP16 | **271.4** | 61.6 |
| `armv8.2_1` | DOTPROD | 270.7 | 54.0 |
| `armv8.0_1` | — | 116.6 | 43.3 |
| `armv9.2_2` / `armv9.2_1` | … **SME** | — | **decline: no SME** |

`armv9.2_1` and `armv9.2_2` are **unmeasured anywhere**: no cloud instance has SME, and Apple M4 has
it but macOS ships one static library rather than the aarch64 module set, so those two modules cannot
even be loaded there.

## The finding across all three machines: the pick costs token generation

This is the result worth carrying upstream, and it is not a one-off — it holds on three different
architectures:

| machine | ggml picks | pp of the pick | best pp | tg of the pick | best tg |
|---|---|---:|---:|---:|---:|
| Granite Rapids | `sapphirerapids` | **550.0** ✓ | 550.0 | 36.0 | **40.9** (`zen4`) |
| EPYC Genoa | `zen4` | 569–595 ✓ | 595.1 (`cascadelake`) | 54.1–55.2 | **60.5** (`haswell`) |
| Graviton 4 | `armv8.6_2` | 198.0 | **271.4** (`armv8.2_2`) | 63.6 ✓ | 63.6 |

`ggml_backend_load_best` scores a module by its **feature level**, not by measured throughput. On
x86 that gets prompt processing right and gives up **10–11 % of token generation**; on aarch64 it
additionally gives up **27 % of prompt processing**. So the richest module is the right default for
prompt-heavy work and the wrong one for generation-heavy work, and nothing in the current design can
express that.

**This is by construction rather than a bug**, which is why there is no patch for it here: overriding
upstream's scoring downstream would be a maintenance burden with no end. It is worth an upstream
issue with these tables (`TODO.md`, "Upstream submissions"), and worth knowing for anyone who cares
about generation latency — `net.ladenthin.llama.backend` cannot select a *CPU* module, so the only
lever today is shipping a directory with one module in it.

## What these tables cannot show

* **One model, one size.** Qwen3-0.6B Q4_K_M only. The quant type decides which kernels run at all —
  AMX has kernels for Q4_0, Q4_1, Q8_0, Q4_K, Q5_K, Q6_K and IQ4_XS — so another quant would move
  the numbers and could move the ranking.
* **4 vCPU.** Token generation is synchronisation-bound, and the Windows OpenMP measurements in
  `CLAUDE.md` show thread scaling can reverse a ranking. A 64-core machine is a different experiment.
* **Prompt processing is one 257-token prompt**, not a 512-token one despite the `pp512` label
  llama.cpp prints; the number is comparable *within* a table, not against upstream's own figures.
* **Nothing about correctness.** Every row that says `ok` produced output; none of them was compared
  token by token against a reference. The C++ suite covers that separately.
