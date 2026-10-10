<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT
-->

# Local test plan

**The rule: this plan covers only what CI structurally cannot do.** Re-running what CI already runs
on every push costs hours and proves nothing, so each section below states why CI cannot answer it.

Results go into `docs/local-test-results-<cpu>.md`, one file per machine, so several machines
accumulate side by side rather than overwriting each other. Numbers in tables, command lines as
code blocks, and a measurement that was not taken is written as *unmeasured* -- never as an
expectation phrased like a result.

## What CI already covers -- do not repeat it here

| Covered by CI, every run | Where |
|---|---|
| the full model-backed Java suite | `test-java-windows-x86_64`, `-msvc`, `test-java-linux-x86_64`, three macOS jobs |
| the C++ suite on x86-64, aarch64, s390x (big-endian, under qemu) | `C++ Tests …`, `build-linux-s390x` |
| every native build of the 27 natives jars, plus their dependency allowlists | the build jobs, `package` |
| the release artifacts launched as `java -jar` | the `smoke-fatjar` matrix, `smoke-agent-linux` |
| the agent's `--web` at HTTP level (401, token cookie, console page) and `--acp` at protocol level | `smoke-agent-linux` |

So a green CI run already means: it builds everywhere, the unit suites pass, the jars load, and the
servers answer. What it does **not** mean is covered below.

## Preconditions, per machine

1. **A JDK whose javac supports `-XDaddTypeAnnotationsToSymbol`.** `llama/pom.xml` passes that flag
   because NullAway's JSpecify mode requires it below JDK 22, and **Oracle JDK 21 does not support
   it** -- `mvn compile` then dies in an Error Prone `IllegalStateException` naming exactly that
   (measured on Oracle 21.0.9+7-LTS-338; the same tree compiles cleanly under Temurin 21.0.12.1+1,
   which is what CI uses via `distribution: temurin`). Check with `mvn -version` that Maven's JDK is
   not an Oracle build before blaming anything else.
2. **The CI model set** in `models/` at the repository root, by the exact filenames of
   `.github/models.csv`. Five of the eleven cannot be substituted by another local model because the
   test needs a *capability*, not a model: the reranker, the embedding model, the vision pair, the
   TTS pair, and `stories260K` (which must be **F32** for the trainer). Every model path has a `-D`
   override (README "System Properties Reference"; `MODEL_PATH`, `DRAFT_MODEL_PATH`,
   `REASONING_MODEL_PATH` and `RERANKING_MODEL_PATH` got theirs -- `net.ladenthin.llama.text.model`,
   `.draft.model`, `.reasoning.model`, `.rerank.model` -- after the first run of this plan), so a
   model that is already on the machine can stand in where the test needs no particular capability.
3. **The toolchain** of the thing being tested: MSVC + Ninja for a Windows CPU build, plain clang for
   a variants build (see `CLAUDE.md` "CPU variants"), the vendor runtime for a GPU build.

## Part A -- which CPU module this machine picks, and what it is worth

**Why CI cannot answer it.** The runners are whatever the cloud provides, so CI can only ever
exercise its own runner's choice. The entire point of the 14-variant build is that different CPUs
load different modules, and that is a property of the hardware, not of the commit.

This is the measurement that justifies the variants feature, and so far only its *floor* is known:
on a Zen 3 the plain `x64` baseline module is **10.4x slower at prompt processing** than `haswell`
(313 -> 30 t/s, Qwen3-0.6B Q4_K_M). The gain *upwards* -- `zen4`'s BF16, `sapphirerapids`' AMX -- is
**unmeasured**.

| Machine | Expected module | Why | Measured? |
|---|---|---|---|
| Ryzen 7 5800H (Zen 3) | `haswell` | AVX2, no AVX-512 | **yes** -- confirmed from the JVM's mapped modules, see the b11538 report |
| **Ryzen AI Max+ 395** (Zen 5) | **`zen4`** | AVX-512 + VNNI + BF16; `sapphirerapids` would need AMX, which Zen 5 lacks | **open -- needs the device** |
| **Core Ultra 7 165U** (Meteor Lake) | **`alderlake`** | AVX2 + AVX-VNNI, no AVX-512 | **open -- needs the device** |

Steps:

1. Build the variant library (Windows needs plain clang; `clang-cl` and `cl.exe` are refused with a
   reason, because ggml's `if (NOT MSVC)` block drops 5 of the 14 variants):
   ```
   .github\build.bat -G "Ninja Multi-Config" -DJLLAMA_CPU_VARIANTS=ON ^
     -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ^
     -DGGML_OPENMP=OFF -DOS_NAME=Windows -DOS_ARCH=x86_64
   ```
2. `python .github/verify-native-deps.py llama/src/main/natives` -- expect every library of the
   directory checked and 0 violations.
3. Run the Java suite against that build (`mvn -f llama/pom.xml test`). A load alone proves little:
   in the Windows measurement **every** crash came after a load that reported the right device count.
4. **Which module the JVM actually mapped.** ggml logs module decisions at `GGML_LOG_DEBUG` only, so
   `-lv 4` may not name it; if it does not, list the DLLs the process has mapped (`listdlls`, Process
   Explorer, `tasklist /m`) rather than inferring from the CPU.
5. `pp512` and `tg128` (Qwen3-0.6B Q4_K_M, 8 threads) for that module, and for the baseline forced
   via `GGML_BACKEND_PATH` or by moving the other modules aside. Both numbers, with the spread.

## Part B -- GPU inference

**Why CI cannot answer it.** GitHub-hosted runners have no GPU at all, so every GPU natives jar is
**build-only forever** -- CUDA, ROCm, SYCL, Vulkan, OpenCL and OpenVINO are compiled and their
dependency lists are checked, and nothing more. Correctness of GPU inference has no CI coverage by
construction.

Per GPU present on the machine: load a model with `-ngl 99`, confirm from the load log that layers
went to the device (the `model buffer size` lines name the backend), run one completion, and compare
its throughput with the CPU backend. Note the backend **by name**, never by device index: indices are
not stable across the set of loaded backends -- measured both ways on the 5800H machine, where
`Vulkan0` is the AMD iGPU and `Vulkan1` the NVIDIA GPU with only the Vulkan backend loaded, the
opposite of the assignment `CLAUDE.md` records from a run with CUDA loaded as well.

Two things to expect, both measured at b11538 (see the report):

* `net.ladenthin.llama.backend` has to **force** the backend under test -- the loader takes
  `cuda13` over `vulkan` over `cpu`, so a machine with several GPU jars only ever exercises the first.
* **Vulkan's first call after a cold shader cache is pathologically slow** (pp512 25.9 against
  ~10 100 once warm). Discard round one, and do not report it as the steady state.

**The GPU backend as a module beside the CPU set** (`-DJLLAMA_CPU_VARIANTS=ON` together with
`-DGGML_CUDA=ON` / `-DGGML_VULKAN=ON`) is the harder question behind "one jar with GPU *and* a full
CPU fallback". It builds and runs -- but two defects stand in the way today, and both are only
visible on a machine with that GPU: the GPU module is built and **not copied** into the natives
directory, and CUDA silently loses two thirds of its token generation because the variants path does
not repeat llama.cpp's `GGML_CUDA_GRAPHS_DEFAULT ON`. Vulkan loses nothing. The report has the
numbers and the one-line cause.

## Part C -- the interactive agent

**Why CI cannot answer it.** `smoke-agent-linux` drives the agent through a pipe: a one-shot prompt,
a tool round, `--web` checked over HTTP, `--acp` spoken by a Python client. The **cursor-controlling
console** is the part that cannot be driven that way -- `llama-atmosphere-agent/CLAUDE.md` records
that it needs someone typing *and* permission to move the cursor, and that `JLineTerminal.open`
deliberately refuses when there is no console (inside a Surefire fork it would seize the channel
Surefire talks over). CI has no TTY and nobody at the keyboard.

Run `mvn compile exec:java -Dexec.args="--model <gguf> --allow-shell"` in
`llama-atmosphere-agent/` and check, in a **real** terminal:

| What | What to look for |
|---|---|
| the pinned block | rule, activity row and state row at the bottom, input directly above it, nothing smeared into the scrollback |
| **dragging the window wider and narrower** | the known-hard case. No staircase of rules, no leftover bar above the region, the conversation reprinted and re-flowed after a width change, the prompt back on its row after a height change (a *shrink* is a recorded open defect -- `/cls` repairs it) |
| `/cls` and Ctrl-L | screen blank, cursor and prompt on the row the block leaves free, scrollback still reachable |
| the approval prompt | `[y]es/[n]o/[a]uto` for `run_command` / `write_file` / `edit_file`; a denial must reach the model as `{"status":"cancelled"}` |
| Shift+Tab | switches `⏸ manual` / `⏵⏵ auto` between turns; the startup line advertises the key only when the bind succeeded |
| typing during a turn | stops the turn, and the typed line becomes the next message |
| the startup line | names the terminal and the JLine patch level -- a screenshot does not say which library produced it, and that cost two rounds of testing before |
| `--web` | in a **real browser**: the token link, then approvals rendered as Approve / Deny, `/stop` cancelling a turn |
| `--acp` | in a **real editor**: tool cards, `session/request_permission`, the slash commands arriving as `available_commands_update` |

Note the JLine version in the report. The repository builds against the **released** JLine, where
`ScreenUseCasesTest` skips itself; the seven carried fixes live in a patched build, and the console
behaves measurably differently with and without them (233 tests: released 4.4.6 **26 red**, the
reviewed set **0**).

## Part D -- the models a patch was written for

**Why CI cannot answer it.** A patch that teaches the library a new architecture is guarded by a C++
test over a **synthesized** GGUF: `test_kolibri1.cpp` writes tiny random models in both published
dialects and compares every logit with an independent double-precision reference. That is the right
guard -- it runs on every platform and fails the build if the patch is dropped -- but it deliberately
proves the *arithmetic*, not the real file. `CLAUDE.md` says so outright for `patches/0016`:
**"Not verified here: the real 78B model … and GPU backends"**, because the session that wrote it had
no HuggingFace access and the smallest GGUF is 28.6 GB. CI has neither the disk nor the time, and
HuggingFace is not a CI dependency anywhere in this repository.

So the real-model check is local by construction, and it is the only thing that can tell a correct
implementation from one that merely agrees with its own reference.

**Not on the 5800H workstation, measured rather than assumed:** 63 GB of RAM against 59.8 GB of
weights plus the context means mmap pages from disk for the whole run, so every number would
describe the disk and not the model. This part is therefore **deferred to a machine with enough
memory to hold the weights** -- roughly 80 GB for Q6_K, or a smaller quantisation on less. The
metadata table below needed no load: it is read from the GGUF header, which is the one part of
this section any machine can do.

**Kolibri-1 (`patches/0016`).** The metadata of the copy on this workstation, read from the GGUF
header alone (no load, no inference), already confirms the architecture the patch describes:

| Key | Value | What it confirms |
|---|---|---|
| `general.architecture` | `kolibri1` | the architecture the patch registers |
| `expert_gating_func` | **5** | the **second** of the two published dialects (the Qwen3-MoE-based port); the patch claims to load this one *and* gating 2 |
| `tokenizer.ggml.pre` | `kolibri1` | the pre-tokenizer name the patch maps to Qwen2 |
| `expert_weights_norm` | 0 | unnormalized routing |
| `expert_count` / `expert_used_count` / `expert_shared_count` | 384 / 6 / **1** | routed MoE plus one ungated shared expert |
| `attention.sliding_window` / `…_pattern` | 513 / a `0` at every fifth of 50 layers | sliding layers interleaved with full-attention (NoPE) layers |
| `attention.head_count` / `…_kv` | 48 / 4 | GQA |
| `context_length` | 262144 | |

What to measure, in this order, because each step is cheap and the next is not:

1. **Does it load?** Both shards (`Kolibri-1-Q6_K-0000{1,2}-of-00002.gguf`, 59.8 GB together). Report
   the architecture line from the load log and whether the pre-tokenizer was accepted. Note the RAM
   of the machine and the drive the file sits on: below the weight size the load time describes the
   disk, not the model.
2. **One short generation, in German** (it is a German/English reasoning model). Coherence is the
   signal: the router selects on `logits + expert_bias` but weights by the *unbiased*
   `sigmoid(logits)`, and a DeepSeek-V3-style router would pick other experts as soon as a bias is
   non-zero -- which produces output that is fluent-looking but wrong. So judge the text, not just
   the absence of a crash.
3. **A GPU backend**, if one has the memory. The 8 GB RTX 3070 cannot hold this model, so a partial
   offload (`-ngl` well below the layer count) is the realistic form; it still exercises the patch's
   graph on a GPU, which is the second thing `CLAUDE.md` records as unverified.

**If the real model disagrees with the synthesized reference, that is a finding about the
implementation, not a test to adjust.** `CLAUDE.md`'s row for `test_kolibri1.cpp` already separates
the two cases: a failing *numerical* comparison means the graph computes something other than Aleph
Alpha's reference; a failing *format* row means upstream's converter chose another spelling. Only the
second is a test to move.

**The same question for any future patch** that adds an architecture: the guard proves the maths, a
real file proves the loader, and only the second needs this machine.

## When to repeat this

| Trigger | Which part |
|---|---|
| a new machine / CPU generation | Part A, and Part B if it has a GPU |
| the variant set or ggml's scoring function changed | Part A |
| a llama.cpp bump that touches `ggml-cpu` | Part A, step 2 and 3 |
| a JLine bump, or a change to the console / front ends | Part C |
| a new GPU backend, or a vendor runtime bump | Part B |
| a change to the variants path's CMake, or a llama.cpp bump touching ggml's option defaults | Part B's module question (the `GGML_CUDA_GRAPHS` class of defect) |
| a patch that adds a model architecture (`0016` today), or a bump that changes one | Part D |

**Not** per commit, and not per release: CI carries the parts that must hold every time.
