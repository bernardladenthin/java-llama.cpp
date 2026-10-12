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

## `GGML_OPENMP=OFF` on Linux costs nothing measurable

The question this answers is a shipping one, not a tuning one: `libggml-base.so` -- the first file the
loader loads -- declares `libgomp.so.1`, and a plain Ubuntu 24.04 server image does not have it, so
the whole Linux jar fails to load there. Turning OpenMP off removes the dependency (ggml then uses
its own `std::thread` pool, which is what the Android build already does). The open question was the
price.

Measured on **one machine, both builds, same source** -- upstream llama.cpp b11538 compiled twice on
an `m7i.4xlarge` (Xeon Platinum 8488C, 16 vCPU, gcc 11.5) with identical flags
(`GGML_BACKEND_DL=ON GGML_CPU_ALL_VARIANTS=ON GGML_NATIVE=OFF`) except `GGML_OPENMP`, both picking
`ggml-cpu-sapphirerapids`, and benchmarked with upstream's own `llama-bench` (warmup plus 3
repetitions, so the figures carry a spread):

| threads | pp512 **with** OpenMP | pp512 **without** | tg128 **with** | tg128 **without** |
|---:|---:|---:|---:|---:|
| 4 | 787.59 ± 7.67 | 784.79 ± 15.12 | 73.76 ± 0.68 | **74.41 ± 0.20** |
| 8 | 1456.02 ± 3.76 | 1425.74 ± 43.73 | 128.57 ± 1.02 | **129.55 ± 1.09** |
| 16 | 1697.60 ± 61.69 | 1669.73 ± 42.49 | 153.49 ± 0.94 | **154.62 ± 0.65** |

**Token generation is slightly faster without OpenMP at every thread count** (+0.9 %, +0.8 %, +0.7 %)
and prompt processing is marginally lower (−0.4 %, −2.1 %, −1.6 %) -- every difference inside or at
the error bars. The `libgomp` dependency is gone in the off build, verified from the binary
(`NEEDED libgomp` count 1 against 0 in `libggml-base.so`).

**So the ~6 % cost this project previously recorded does not reproduce.** That figure was one sample
with its own ON value at 76.1 ± 0.9 on a different machine, and it was already flagged as "inside the
spread rather than a result"; with three thread counts and repetitions there is no cost to weigh
against the dependency at all.

**Limits, so the number is not over-read:** gcc 11.5 (what Amazon Linux 2023 ships), not the
gcc-toolset-14 CI builds with; one CPU; one model; upstream's build rather than this project's jar
(the flag is a ggml-level one, so it transfers, but the jar also carries llama.cpp, mtmd and the
server). The direction is unambiguous across three thread counts, which is what the decision needed.

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

## Re-measured at b11558 on 64 cores: the baseline moved, and the ranking reverses

Measured **2026-10-12** against the **published** jars (`5.2.0-20261012.112736-13`, llama.cpp
**b11558**) on an `m8i.16xlarge` -- the **same CPU model** as the table above (Xeon 6975P-C, Granite
Rapids) but **64 vCPU instead of 4**. Same method: one module staged at a time, same model, 35 runs in
one boot.

This exists because the section below predicted it: the b11538 tables are relative to `haswell`, and
**upstream #30270 (merged 2026-10-11, in the b11538..b11556 range) makes AVX2+F16C CPUs use the
hardware conversion instead of a lookup table** -- i.e. it moves exactly that baseline.

### All 14 modules at 16 threads

| module | pp512 | tg128 | against `haswell` (pp) |
|---|---:|---:|---:|
| **`sapphirerapids`** (AMX) ← ggml's choice | **2775.5** | **160.0** | **+96 %** |
| `cooperlake` | 1949.1 | 152.5 | +37 % |
| `skylakex` | 1801.4 | 153.0 | +27 % |
| `cannonlake` | 1787.7 | 151.9 | +26 % |
| `icelake` | 1565.7 | 152.9 | +10 % |
| `cascadelake` | 1563.1 | 152.8 | +10 % |
| `zen4` | 1553.2 | 153.9 | +10 % |
| `haswell` | 1417.6 | 151.8 | — |
| `alderlake` | 1410.0 | 150.4 | −1 % |
| `piledriver` | 892.6 | 140.6 | −37 % |
| `ivybridge` | 861.2 | 138.3 | −39 % |
| `sandybridge` | 841.2 | 76.2 | −41 % |
| `sse42` | 68.9 | 36.4 | −95 % |
| `x64` | 63.4 | 33.2 | −96 % |

Full set on the same machine: ggml chose `sapphirerapids`, 2763.4 / 159.6 -- within noise of the
staged run, so the staging still does not distort anything. **All 14 load and run**, as at b11538.

**AMX is worth +96 % on prompt processing here, against +148 % in the 4-vCPU table.** Two things
changed at once (the pin and the machine size), so see the isolation attempt below before reading that
as the effect of #30270 alone.

### The thread sweep: at 64 threads `haswell` beats the module ggml picks

The open question was whether the generation ranking reverses with thread count, as the Windows
OpenMP measurements showed it can. **It does.** Token generation, four modules, same machine:

| threads | 1st | 2nd | 3rd | 4th |
|---:|---|---|---|---|
| 4 | **`sapphirerapids`** 59.7 | `zen4` 55.2 | `cascadelake` 54.1 | `haswell` 53.8 |
| 8 | **`sapphirerapids`** 105.6 | `cascadelake` 102.0 | `zen4` 101.5 | `haswell` 99.4 |
| 16 | **`sapphirerapids`** 159.1 | `zen4` 153.1 | `cascadelake` 151.7 | `haswell` 150.7 |
| 32 | **`sapphirerapids`** 188.2 | `cascadelake` 186.8 | `haswell` 185.1 | `zen4` 183.8 |
| **64** | **`haswell` 192.6** | `zen4` 190.0 | `cascadelake` 188.6 | **`sapphirerapids` 181.8** |

At 64 threads the richest module is **last**, and the one ggml would never pick on this CPU is first.
Prompt processing turns over there too: between 32 and 64 threads `sapphirerapids` falls 4119.7 → 3880.2
and `zen4` collapses 2069.0 → **1143.3**, while `haswell` keeps rising 1752.8 → 1960.3. The machine has
32 physical cores with SMT, so 64 threads means hyperthreads -- which is a realistic default, since
`std::thread::hardware_concurrency()` and most "use all cores" settings report 64 here.

**This sharpens the upstream finding below rather than adding a new one**: `ggml_backend_load_best`
scores by feature level, which costs 10-11 % of generation at 4 vCPU and **loses outright at 64
threads**.

### #30270 is NOT the reason the baseline moved -- measured, and it refutes what this page first said

Same CPU model and the **same thread count** (4), the only pair comparable across the two tables:

| | b11538, 4 vCPU | b11558, 64 vCPU at t=4 | change |
|---|---:|---:|---:|
| `haswell` pp512 | 221.8 | 422.7 | **+91 %** |
| `sapphirerapids` pp512 | 550.0 | 847.9 | +54 % |
| ratio | 2.48x | **2.01x** | |

This page first read that asymmetry as *consistent with #30270*, since `haswell` gained far more than
the AMX module and #30270 is an AVX2-specific fix. **That was wrong, and the clean isolation says so**
([`upstream-f16c-30270.md`](upstream-f16c-30270.md)): one source tree built twice on one machine with
those eight lines removed shows **no measurable effect at all** -- every difference inside ±2 %, with
two non-F16C control modules moving just as much.

So the +91 % is **the instance size**: 4 threads on a 64-vCPU machine have far more memory bandwidth
to themselves than a 4-vCPU machine does, and `haswell` is the more bandwidth-bound of the two. The
asymmetry between two rows of two different machines is not an isolation, and reading it as one was
the mistake. It is kept here rather than quietly removed, because the next person comparing two tables
across a pin **and** a machine size will be tempted the same way.

## These numbers are bound to llama.cpp b11538, and b11556 already moved one of them

Every table above was measured against the natives jars built at **b11538**. That is not boilerplate:
**upstream #30270, in the b11538-to-b11556 range, fixes F16C use on AVX2 CPUs** -- i.e. exactly the
`haswell` module class, which is the baseline every "against `haswell`" column here is computed
against, and the module a Zen 3 workstation and an EPYC Milan instance both select. fp16 conversion
throughput there is therefore not the same before and after the bump.

So: **do not compare a figure measured at b11556 or later with one from this page.** Re-measure the
baseline in the same run, which the harness makes cheap -- every table here was produced by staging
the modules on one machine in one boot, and that is what keeps a comparison honest. When you do,
record the tag next to the numbers as these tables do, and add a row rather than editing one: a table
that silently mixes two llama.cpp versions is worse than two tables that disagree.

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
