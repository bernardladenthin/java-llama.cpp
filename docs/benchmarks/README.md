<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Benchmarks and hardware measurements

Numbers this project measured on real hardware, with enough context that each one can be re-read and
re-run years later. Everything here is a *measurement*, not a plan — the plan for what still needs
measuring is [`../rented-hardware-test-plan.md`](../rented-hardware-test-plan.md), and the open items
themselves live in [`../../TODO.md`](../../TODO.md).

| File | What it answers |
|---|---|
| [`cpu-backend-modules.md`](cpu-backend-modules.md) | What each of the 22 shipped CPU backend modules is worth, per microarchitecture — and whether `ggml_backend_load_best` picks the fastest one |
| [`moe-mxfp4-cpu-alderlake.md`](moe-mxfp4-cpu-alderlake.md) | Where the `alderlake` module (Core Ultra) loses time on Qwen3 and gpt-oss-20b, measured with random-weight models in the real shapes -- and what the fix to ggml's MoE path in `llama/patches/0022` is worth (+47 % prompt, +26 % generation; switchable for A/B) |
| [`moe-repack-real-weights.md`](moe-repack-real-weights.md) | `llama/patches/0022` with trained weights (gpt-oss-20b MXFP4, Qwen3-30B-A3B Q4_K_M, OLMoE Q4_0) on Zen 3, a Neoverse-N1 and an IBM z16: +13 % to +66 % prompt processing, generation unchanged, KLD against upstream's path -- and the ARM defect of the first version it found |
| [`gpu-module-jars.md`](gpu-module-jars.md) | What happens when a GPU module jar is actually *used* -- two findings that contradict the "a module that cannot help is skipped" premise |
| [`arm-fp16-conversion.md`](arm-fp16-conversion.md) | Every shipped ARM binary converts fp32 to fp16 in software, ours and upstream's: 26 of 26 disassembled, 8.9x to 13.3x on the function, and 0.03 % of cycles so inference does not move |
| [`upstream-f16c-30270.md`](upstream-f16c-30270.md) | An isolation of one upstream commit: what it does, that it changes nothing measurable here, and a compile-time guard proven red against the defect |
| [`native-loading.md`](native-loading.md) | What extracting the natives costs and what the reuse saves, how a running JVM's locked files behave, and which container base images can load the Linux jar |
| [`ec2-harness.md`](ec2-harness.md) | How a measurement is run on rented hardware, and the invariants that came out of getting it wrong |

## Why these exist at all

The project ships **14 x86-64 and 8 aarch64 CPU backend modules**, of which ggml keeps exactly one at
run time — the best for the running CPU. A module is therefore only ever exercised on a machine whose
CPU prefers it, and no amount of running the test suite on one machine or on a CI runner changes that.
Two consequences shaped everything in this directory:

* **A number from one machine says nothing about another.** The only comparison that means anything
  is one module against another *on the same machine with the same model*, which is how every table
  here is produced.
* **CI cannot do this.** GitHub mixes runner CPUs, so which module a run exercises is chance — a
  property that did not merely limit benchmarking but **hid a crash for months** (see the AMX row of
  `CLAUDE.md`'s patch table: the Windows job died only on the runs that happened to land on AMX
  hardware).

## House rules for adding a measurement

1. **Name the machine, the model, the thread count and the build.** A table without them cannot be
   compared with anything later.
2. **Vary one thing.** Every table here stages one module at a time with everything else held fixed;
   a comparison across two machines is a different question and is labelled as such.
3. **Record the failures too.** A module that declines to load is a *result* — it is how the CPU
   feature gate is supposed to work — and the exact wording matters, because `no CPU backend found`
   is correct behaviour while a crash is not.
4. **Say what the run cannot show.** Every table here carries its own limits; a reader who does not
   know them will over-read it.
