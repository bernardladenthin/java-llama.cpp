<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Test plan for rented hardware

A work queue for the questions this project cannot answer on CI or on a developer machine, to be
worked through one rented instance at a time. The companion document for a local machine is
[`local-test-plan.md`](local-test-plan.md); the open items themselves live in
[`../TODO.md`](../TODO.md), and this file is only the *how and in which order*.

**What has already been measured is in [`benchmarks/`](benchmarks/)** --
[`benchmarks/cpu-backend-modules.md`](benchmarks/cpu-backend-modules.md) for the per-module numbers
and [`benchmarks/ec2-harness.md`](benchmarks/ec2-harness.md) for the mechanics of a run.

**The account-specific mechanics are deliberately not here** — how an instance is launched, how a
result comes back, which bucket and which credentials. Those are private notes, because they are
account data rather than project knowledge.

## Why rented hardware answers anything at all

CI already runs the whole suite on every platform, so renting a machine is only worth it where CI is
structurally blind:

| CI cannot | because |
|---|---|
| choose a CPU | GitHub mixes runner hardware; which feature set a run draws is chance. **This is not theoretical:** the AMX heap corruption (`patches/0018`) showed up as `Tests run: 0` on the runs that happened to land on Sapphire Rapids or newer, and a green Windows job proves nothing unless its log names an AMX CPU |
| use a GPU | every hosted runner is GPU-less, so the GPU module jars are only ever *placed* and skipped, never loaded |
| hold a large model | a 78B GGUF needs more RAM than a runner has, and more disk than the cache allows |
| run for an hour | the throughput questions need sustained load, not a build step |

A developer machine answers the first and last but has exactly one CPU and one GPU, which is the
whole problem: `ggml` has 14 x86 and 8 aarch64 CPU modules, and a machine proves one of them.

## The invariants, learned the expensive way

These are not style preferences; each one comes from a run that went wrong.

1. **Always terminate, never stop.** A terminated instance costs nothing afterwards; a stopped one
   keeps paying for its disk. Nothing here needs to survive between runs.
2. **A dead-man switch is the first line of every script** (`shutdown -h +<minutes>`, with the
   instance set to terminate on shutdown). Without it, a script that hangs runs until something
   external kills it — and nothing does when the watching process on the developer machine dies.
   With it, the worst case is bounded by one number you choose in advance.
3. **Many checks per boot.** Booting, installing a JDK and fetching a model is 10–15 minutes; the
   measurement is often seconds. Stage every variant in one script and vary only the thing under
   test.
4. **One variable at a time, on the same machine.** What made the AMX finding conclusive was staging
   one CPU module at a time with the same model on the same instance: the module was the only
   difference left. A comparison across two instances proves much less.
5. **Stream the result out after every step**, to a destination with a name unique to that run. A
   run-specific name is not pedantry — reusing a result file twice made a watcher read the *previous*
   run's finished log, declare success and terminate a running instance mid-measurement.
6. **Read the error lines, not the exit code.** A probe whose every run exits non-zero looks like
   "the thing under test is broken" and is usually the harness: a mistyped path, an unexpanded
   placeholder, a missing file. Print the output.
7. **Fetch public models on the instance, not through your own connection.** A rented machine
   downloads from Hugging Face at datacentre bandwidth. Only stage something yourself when it is not
   publicly available.

## The queue

Ordered by what each one unblocks, not by cost. Every row ends with *what counts as an answer*, so a
run cannot end in "looked fine".

### 1. The GPU module jars on a real GPU

**The question.** Modular natives (6.0.0) made each GPU backend its own module jar, loaded next to
the platform's CPU library. CI places them on a GPU-less runner, where `ggml` skips them — so the
*successful* path has never run. Does a `cuda13` or `vulkan` module jar actually register its device,
offload layers and compute, next to the CPU library it was built beside?

**How.** The cheapest NVIDIA instance class, a small CPU-only-capable model, a driver-equipped image
so no toolkit install is needed. Five runs in one boot: automatic backend selection, the GPU backend
forced with full offload, the same forced with zero offload, a second GPU backend forced, and the
CPU library alone as the baseline. Log verbosity high enough that the device enumeration and the
`offloaded N/N layers` line appear.

**An answer looks like:** the loader line names the module, `ggml` reports the device, the offload
line shows all layers, and generation is materially faster than the CPU baseline. A module that
silently falls back to the CPU is a *finding*, not a pass.

**Also re-measures:** whether the CUDA module still carries `GGML_CUDA_GRAPHS` — a build that lost
that default once cost two thirds of its token generation, and only a GPU can see it.

### 2. Kolibri-1 against the real 78B model

**The question.** `patches/0016` adds an architecture upstream does not support, and it is verified
only against tiny synthesized GGUFs. Two community conversions exist whose GGUFs are mutually
incompatible, and the patch claims to load **both**.

**How.** One high-RAM instance, both dialects in a single boot, each GGUF fetched from its repository
on the instance. Take a middle quant rather than the smallest: the question is whether the German
answer is *coherent*, and a heavily quantized 78B MoE would make that ambiguous. Print the
architecture, expert-count and gating lines from the load, then the generated **text** of a raw
completion and of a chat turn through the model's own template.

**An answer looks like:** both dialects load, and the German answer is coherent. Fluent-looking
nonsense is the specific failure to watch for — the router selects on `logits + expert_bias` but
weights by the *unbiased* sigmoid, and getting that wrong produces text that reads well and is wrong.

### 3. The throughput gain of the top CPU modules

**The question.** The CPU-variant work proved the *lower* bound (the baseline `x64` module is ~10×
slower at prompt processing than `haswell`). The gain upwards — `zen4` with BF16, `sapphirerapids`
with AMX — is unmeasured, which is why the default build ships every level.

**How.** One instance per microarchitecture (Granite Rapids, EPYC Genoa, and an older generation as
the `haswell` reference), identical model, identical thread count, each staging its own module plus
`haswell` for a same-machine comparison.

**An answer looks like:** a table of pp512 and tg128 per module per machine. **Beware the result
already in hand:** on Graviton 4, `ggml`'s own choice was 37 % *slower* at prompt processing than a
module it rejected, so measure every module the CPU accepts, not only the selected one.

### 4. The OpenVINO module, with no GPU at all

**The question.** `openvino-linux-x86-64` is build-only; nothing has ever loaded it. OpenVINO has a
CPU plugin, so this needs no GPU and no quota — the cheapest open item in the matrix.

**An answer looks like:** the module registers a device and computes, or it fails with a reason worth
recording.

### 5. Extraction reuse on Windows

**The question.** The extraction directory is keyed by build and module set and reused across JVM
starts. A second start of the same build must copy nothing, a running JVM's locked libraries must
survive another build's cleanup, and a start must not block on them. This is Windows-specific
(file locking) and needs two JVMs.

**An answer looks like:** no extraction lines on the second start, and both JVMs healthy.

### 6. The full Java suite on a rented platform

Worth doing once the staging is in place anyway: the whole model-backed suite against a shipped
natives artifact, on a platform or CPU feature set CI cannot pin. Lower priority than the above
because CI already runs this suite on every platform — what rented hardware adds is *which* CPU.

## What rented hardware cannot answer

Recording the limits so they are not re-derived per attempt:

| open item | why not |
|---|---|
| `armv9.2_1` / `armv9.2_2` (the two **SME** aarch64 modules) | no cloud instance has SME. Apple M4 does, but macOS ships one static library rather than the aarch64 module set, so those modules cannot even be loaded there |
| **ROCm** at runtime | the only AMD-GPU instance class has been withdrawn, and ROCm never officially supported that card anyway. Needs a real Radeon machine |
| **SYCL** on a GPU | no discrete Intel GPU is offered. Intel's CPU OpenCL runtime covers part of the path |
| **Metal** | macOS instances bill a 24-hour minimum, and Metal is compiled into the library there, so CI's macOS jobs already cover it |
| a real **arm64 Android device** | the AAR is exercised on an x86-64 emulator in CI; the Adreno/OpenCL flavor needs physical hardware |
| the **interactive agent** | the terminal agent's console behaviour — line editing, the status line, interrupting a stream — needs a human at a real terminal. This is the one item that stays with a developer machine |

## Cost discipline

The honest framing: these runs cost **cents to low single-digit euros each**, and the risk is never
the hourly rate but a forgotten resource. Three habits cover it:

* the dead-man switch above, so an instance cannot outlive its purpose;
* a **sweep over every region** after a run — not just the obvious one, and not only for instances: a
  disk can **survive a terminate** and keeps costing money with no instance attached, which an
  instance query does not show. Snapshots and reserved addresses are the same class;
* a **budget alert** on the account, which is the only guard that reaches a human whether or not
  anything is watching.

A refused launch — a quota not yet raised, no capacity for that instance type — costs nothing, so a
retry loop may poll for hours without consequence.
