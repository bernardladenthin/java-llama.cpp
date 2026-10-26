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

> ### Resolved, and re-measured on the same hardware (2026-10-12)
>
> The module now ships **native code for `sm_75`** instead of PTX only, so the driver has nothing to
> JIT and the abort cannot occur. Verified by re-running the identical probe: **same AMI, same driver
> 595.91.07, same Tesla T4 — only the module changed.**
>
> | run | before (PTX only) | after (`sm_75` cubin) |
> |---|---|---|
> | automatic, `-ngl 99` | **exit 134** | **exit 0** · 29/29 layers · pp **4969.0** / tg **232.0** |
> | `cuda13` forced, `-ngl 99` | **exit 134** | **exit 0** · 29/29 · pp **4973.4** / tg **228.3** |
> | `cuda13` forced, `-ngl 0` | **exit 134** | **exit 0** · 0/29 · pp 1244.3 / tg 31.4 |
> | `vulkan` forced (control) | exit 0 · 201.8 / 24.1 | exit 0 · 240.3 / 32.6 |
> | `cpu` only (baseline) | exit 0 · 202.2 / 24.1 | exit 0 · 240.9 / 33.3 |
>
> Three aborts became three clean runs, and the T4 delivers **20.6x prompt processing** and **7.0x
> generation** over the CPU baseline on the same machine. That `-ngl 0` runs again closes the case:
> it was the most absurd of the three, since not one layer was on the GPU.
>
> **What the shipped module carries now**, read with `cuobjdump` from the published jar:
>
> | | architectures |
> |---|---|
> | native code (cubin) | `sm_75` `sm_80` `sm_86` `sm_89` `sm_90` `sm_100` `sm_120a` `sm_121a` — 144 each |
> | PTX (the JIT reserve) | `sm_121a`, 144 files |
>
> So Turing, A100 and Hopper no longer depend on the driver, and anything newer than Blackwell still
> has PTX to fall back on. The `a` suffixes are upstream's own doing rather than a defect in the list
> this project passes: `ggml/src/ggml-cuda/CMakeLists.txt` rewrites every `12x` architecture to
> `12xa`, deliberately, because `120a-real` works with any CMake version.
>
> **One method note, because it nearly produced a wrong conclusion:** a first scan reported *no* PTX,
> because it searched for `compute_`. `cuobjdump` names PTX files `libggml-cuda.N.sm_121a.ptx` — with
> `sm_`. Read the tool's raw output before concluding from a pattern.
>
> The original finding follows, unchanged, as the record of what the defect was.

### The defect as it was found


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

**And there is no safe configuration.** The same machine, the same module, five runs:

| run | layers offloaded | result |
|---|---|---|
| loader decides, `-ngl 99` | 29/29 | **exit 134** (`SIGABRT`) |
| `cuda13` forced, `-ngl 99` | 29/29 | **exit 134** |
| **`cuda13` forced, `-ngl 0`** | **0/29** | **exit 134** |
| `vulkan` forced | 29/29 *(log line only)* | exit 0, pp 201.8 / tg 24.1 |
| `cpu` only | 0/29 | exit 0, pp 202.2 / tg 24.1 |

`-ngl 0` aborts too, so this is not about offloading: **having the module on the classpath is
enough**, because the backend registers and llama.cpp queries and uses the device during setup
whatever the layer count says. The only escape is keeping the jar off the classpath or
`-Dnet.ladenthin.llama.backend=cpu`.

The Vulkan row is the control, and it is the one backend of the three that behaved exactly as
designed: skipped for want of an ICD, with throughput **identical to the CPU baseline**
(201.8/24.1 against 202.2/24.1). Note that the `offloaded 29/29 layers to GPU` line appears for it
anyway -- llama.cpp logs the intent, not the outcome, as it also does for a registered RPC device
that holds no layers.

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
* **Two candidate fixes** were weighed: document a minimum driver version for the `cuda13` jars,
  and/or pass an explicit arch list adding `75-real 80-real 90-real` so the common datacenter GPUs
  do not depend on JIT at all. The second was taken: both CUDA builds pass such a list (CLAUDE.md,
  "Upgrading CUDA Version"). The second costs binary size — the module is
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

### Re-measured 2026-10-12, and the README's figures were too kind

On an `m7i.xlarge` (Xeon Platinum 8488C, Sapphire Rapids, 4 cores, no Intel accelerator), Ubuntu
24.04, the OpenVINO 2026.4.1 archive installed exactly as the CI job installs it, against the
**published** jars:

| run | OpenCL ICD | backends registered | pp512 | tg128 |
|---|---|---:|---:|---:|
| `openvino` named, **no ICD** | absent | **2** | 497.2 | 36.4 |
| `openvino` named, **ICD installed** | present | **3** | **22.2** | **0.1** |
| jar on the classpath, property **unset** | present | **2** | 502.9 | 36.2 |
| `cpu` named (baseline) | present | 2 | 481.3 | 33.6 |

Against the baseline on that machine the cost is **21.7x on prompt processing and 336x on
generation** — worse than the 17x/172x this page recorded from the earlier machine, so the README's
numbers understate it. Generation falls to **0.1 tokens per second**, roughly ten seconds per token,
with 27,639 log lines mentioning OpenVINO.

Two things this settles that the earlier run could not:

* **Without the ICD the module does not open at all.** The loader places it (`with GPU module(s)
  openvino`) and ggml reports two backends, with throughput identical to the CPU baseline. The
  OpenVINO runtime archive does not ship `libOpenCL.so.1`; `ocl-icd-libopencl1` is what changes the
  answer, and installing it is what makes the regression appear.
* **The opt-in holds.** With the jar on the classpath and `net.ladenthin.llama.backend` unset, the
  loader says `is on the classpath but not used`, two backends register and the machine runs at full
  speed. This is the **first proof on hardware where OpenVINO could genuinely have taken over** — in
  CI there is no Intel accelerator, so "nothing happened" proves nothing there.

### What to do about it


Decided as recommended: **the OpenVINO module is opt-in** rather than additive
(`LlamaLoader.OPT_IN_BACKENDS`, the `optin` column of `natives.csv`). The premise of additive module
jars is "a module that cannot help is skipped, one that can help is used" — OpenVINO satisfies
neither on a CPU-only host. The README says that the jar needs an Intel GPU/NPU *and* an OpenCL ICD
loader.

## 3 · `cuda13` and `vulkan` together on one NVIDIA GPU: deduplicated, CUDA wins

Measured **2026-10-26** on a Windows laptop with an NVIDIA RTX 3070 Laptop GPU (8 GB, driver 581.83,
CUDA 13.3 toolkit on `PATH`) and an AMD Radeon iGPU. The jars were the **published 6.0.0** set:
`llama`, `cpu-windows-x86-64`, `cuda13-windows-x86-64` and `vulkan-windows-x86-64`. The test ran the
embedded server on JDK 21, with `Qwen2.5-1.5B-Instruct-Q4_K_M` at `-c 4096 -ngl 99 -lv 4` and three
identical chat requests of 128 tokens each (`cache_prompt: false`). `-Dnet.ladenthin.llama.backend`
selected the set without swapping jars.

The question came from the post-release check of 6.0.0. The registry lists the discrete GPU twice,
as `CUDA0` and as a Vulkan device. Does llama.cpp use it twice, or once, and through which backend?
At the default verbosity the answer was filtered out of the log; at `-lv 4` it is printed:

```
load_backends_next_to_this_library: 4 backend(s) registered [CUDA, RPC, Vulkan, CPU], 4 device(s): CUDA0 (CUDA), Vulkan0 (Vulkan), Vulkan1 (Vulkan), CPU (CPU)
llama_prepare_model_devices: skipping device Vulkan0 (NVIDIA GeForce RTX 3070 Laptop GPU) with id 0000:01:00.0 - already using device CUDA0 (NVIDIA GeForce RTX 3070 Laptop GPU) with the same id
llama_prepare_model_devices: using device CUDA0 (NVIDIA GeForce RTX 3070 Laptop GPU) (0000:01:00.0) - 7116 MiB free
load_tensors: offloaded 29/29 layers to GPU
load_tensors:        CUDA0 model buffer size =   934.70 MiB
```

| Set (`-Dnet.ladenthin.llama.backend`) | Device used | Prompt t/s (requests 2, 3) | Generation t/s (requests 1–3) |
|---|---|--:|--:|
| `cuda13` | `CUDA0` | 2357, 2370 | 186.5, 193.5, 187.7 |
| `vulkan` | `Vulkan1` (the RTX) | 2134, 2120 | 193.6, 189.3, 188.7 |
| both (unset) | `CUDA0`; `Vulkan0` skipped | 2571, 2426 / 2231, 2013 | 185.9, 194.6, 190.9 / 179.7, 190.8, 192.6 |

The first request of each server is left out of the prompt column: it includes warm-up (950–1400 t/s).
The two "both" rows are two separate server runs.

- **Deduplication happens in device selection, by PCI id, and the first registered backend wins.**
  CUDA registers before Vulkan, so the model goes to `CUDA0`. No VRAM is spent twice, and nothing is
  split between two handles of the same GPU.
- **The iGPU is not drafted in.** Only the discrete GPU is used, with Vulkan alone as well as with
  both modules. That agrees with the earlier throughput pair on this host: 4 devices at 92.0 t/s
  against cuda13 alone at 94.3 t/s.
- **CUDA and Vulkan are equal on this model** (~190 t/s generation, within noise). The 1.5B model is
  small; a larger one may separate them. The tg128 comparison against the static build's 265.6 t/s
  in TODO.md used a different model and is not answered by this table.
- **OpenCL does not take the NVIDIA GPU at all.** With `opencl-windows-x86-64` added as a third
  module, ggml's OpenCL backend selects the "NVIDIA CUDA" platform, then logs `unsupported GPU
  'NVIDIA GeForce RTX 3070 Laptop GPU'` and `drop unsupported device`. It registers as a backend with
  no device: 5 backends, still 4 devices. The model goes to `CUDA0` as before (generation 182–190
  t/s), because ggml's OpenCL backend targets Qualcomm Adreno and Intel GPUs.
- **Vulkan's device order depends on the modules loaded.** With Vulkan alone, `ggml_vulkan: 0` is the
  AMD iGPU and `1` the RTX. With CUDA loaded as well, `0` is the RTX and `1` the iGPU. A device name
  such as `Vulkan1` therefore names a different GPU depending on which module jars are on the
  classpath; the README says so next to the module list.

## What these runs did *not* test

* **No Intel GPU or NPU was available**, so nothing here says what OpenVINO does on the hardware it
  is for. The 172× figure is specifically the CPU-plugin fallback.
* **Vulkan was not exercised**: the Deep Learning AMI provided no Vulkan ICD, so that module was
  skipped — which is the documented graceful path, and the only one of the three that behaved as
  designed.
* **One GPU, one driver, one model.** A newer driver very likely fixes the CUDA abort; that is the
  point of documenting a floor rather than calling the jar broken.
