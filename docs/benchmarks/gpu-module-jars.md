<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# The GPU module jars on real hardware — two findings that break a design assumption

Measured **2026-10-11** against the natives jars CI built at llama.cpp **b11538**, on rented
hardware. These were the first runs in which a GPU module jar was ever *used* rather than merely
placed: CI has no GPU, so until now the only path exercised was "ggml skips the module and the model
runs on the CPU".

**The assumption under test**, quoted from `CLAUDE.md`'s "Natives jars" section, and the whole
premise of additive module jars:

> A module whose vendor runtime is missing fails to open -- silently inside ggml […] -- and the model
> runs on the CPU modules.

**Both measurements contradict it**, in different ways. Neither is a defect in this project's code;
both are things a consumer will hit, and neither is currently documented.

## 1 · `cuda13` on a Tesla T4: it does not fall back, it aborts the JVM

`g4dn.xlarge`, Tesla T4 (compute capability **7.5**, Turing), NVIDIA driver **595.91.07** (the AWS
Deep Learning Base AMI), the `cuda13-linux-x86-64` module jar next to the modular Linux CPU library.

What worked, and it is worth stating because it answers the open item: **the loader placed the
module, ggml opened it, enumerated the GPU and offloaded every layer.**

```
devices: found 1 CUDA devices · Device 0: Tesla T4
offload: offloaded 29/29 layers to GPU
```

Then:

```
CUDA error: the provided PTX was compiled with an unsupported toolchain.
ggml-cuda.cu:109: CUDA error
libggml-base.so(ggml_abort+0x11e)
exit=134            # SIGABRT
```

**The JVM is killed.** Not an exception, not a fallback to the CPU modules — `ggml_abort` calls
`abort()`, which no Java-side guard can intercept.

### Why the T4 depends on JIT at all

ggml chooses the architecture set itself (`ggml/src/ggml-cuda/CMakeLists.txt`), and this project
**deliberately passes no `CMAKE_CUDA_ARCHITECTURES`** so that the shipped jar gets upstream's full
set. For CUDA ≥ 13 that set is, in CMake's notation:

| architecture | kind | GPUs |
|---|---|---|
| `75-virtual` | **PTX only** | Turing — T4, RTX 20xx |
| `80-virtual` | **PTX only** | A100 |
| `86-real` | cubin | A10/A10G, RTX 30xx |
| `89-real` | cubin | L4, L40S, RTX 40xx |
| `90-virtual` | **PTX only** | H100 |
| Blackwell (`120`/`121`) | cubin | RTX 50xx, B200 |

A scan of the shipped 154 MB module finds the strings `sm_86`, `sm_89`, `sm_120`, `sm_121` and none
for 75, 80 or 90 — consistent with those three being *virtual*, since nvcc stores PTX compressed
inside `.nv_fatbin` where a plain-text scan cannot see it. The run itself is the proof that PTX for
the T4 *was* present: the driver reports it **found** PTX and refused it, which is a different error
from "no kernel image is available for execution".

So on Turing, Ampere-A100 and Hopper the module ships **no native code** and depends entirely on the
driver JIT-compiling PTX produced by **CUDA 13.4**. A driver older than that toolkit cannot, and
driver 595.91.07 could not.

### What this means, and what to do about it

* **The documented invariant "every artifact runs on every GPU generation" is not true as stated.**
  It runs natively on `sm_86`, `sm_89` and Blackwell; on Turing, A100 and Hopper it runs only with a
  driver new enough to JIT CUDA 13.4 PTX.
* **The failure mode is the real problem.** A missing vendor runtime is handled gracefully (the
  module does not open); a *present but too old* driver aborts the process. For a library that is
  loaded inside someone else's JVM this is the worst possible outcome, and it cannot be caught
  downstream.
* **Two candidate fixes**, both for the owner to weigh (`TODO.md`): document a minimum driver version
  for the `cuda13` jars, and/or pass an explicit arch list adding `75-real 80-real 90-real` so the
  common datacenter GPUs do not depend on JIT at all. The second costs binary size — the module is
  already 154 MB — and is exactly the trade-off upstream made the other way.

## 2 · `openvino` on a CPU-only host: it loads, takes over, and is 172× slower

`m7i.xlarge` (Xeon Platinum 8488C, **no Intel GPU**), Ubuntu 24.04, the official OpenVINO 2026.4.1
runtime archive staged exactly as the CI job does, `openvino-linux-x86-64` next to the CPU library.
Same model, same machine, same threads; only the classpath differs.

| configuration | backends registered | pp512 | tg128 |
|---|---|---:|---:|
| `-Dnet.ladenthin.llama.backend=cpu` (module filtered out) | 2 | **490.4** | **34.5** |
| OpenVINO module present, loader decides | **3** | 29.4 | **0.2** |
| OpenVINO forced | 3 | 29.6 | 0.2 |

**17× slower prompt processing and 172× slower token generation** — 5 seconds per token. The run
produced over 27 000 log lines mentioning OpenVINO, which is what a per-operation fallback looks
like. Inference *succeeds*; it is simply unusable.

This is the case the sentence above does not cover. It is written for a module that cannot open at
all; OpenVINO's CPU plugin opens fine and **does** contribute a device, so llama.cpp's device
selection prefers it, and a consumer who merely puts the jar on the classpath gets a silent
catastrophic regression on any machine without an Intel accelerator.

### And it needs a library nobody ships

`libggml-openvino.so` declares `libopenvino.so.2641`, `libtbb.so.12` and **`libOpenCL.so.1`**. The
first two come with the OpenVINO archive; **the OpenCL ICD loader does not**, and it is not installed
by default on Ubuntu (`ocl-icd-libopencl1`). Without it the module silently fails to open — the first
run of this measurement showed exactly 2 backends and identical CPU throughput — so a user can
install everything OpenVINO documents and still get nothing.

`CLAUDE.md` names this dependency **for Windows** (`jllama-extras.txt` carrying OpenVINO's
`OpenCL.dll`), where the OS supplies the loader anyway. For Linux it is undocumented.

### What to do about it

Recorded in `TODO.md` as the owner's call, with a recommendation: **make the OpenVINO module opt-in**
rather than additive. The premise of additive module jars is "a module that cannot help is skipped,
one that can help is used" — OpenVINO satisfies neither on a CPU-only host. Failing that, the README
needs to say that the jar requires an Intel GPU/NPU *and* an OpenCL ICD loader, and that it must not
be on the classpath otherwise.

## What these runs did *not* test

* **No Intel GPU or NPU was available**, so nothing here says what OpenVINO does on the hardware it
  is for. The 172× figure is specifically the CPU-plugin fallback.
* **Vulkan was not exercised**: the Deep Learning AMI provided no Vulkan ICD, so that module was
  skipped — which is the documented graceful path, and the only one of the three that behaved as
  designed.
* **One GPU, one driver, one model.** A newer driver very likely fixes the CUDA abort; that is the
  point of documenting a floor rather than calling the jar broken.
