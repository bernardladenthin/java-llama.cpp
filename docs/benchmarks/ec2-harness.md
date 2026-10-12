<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Running a measurement on rented hardware (EC2)

How the numbers in this directory were produced, written down so a later session can repeat one
without rediscovering the mechanics. The *account-specific* parts — which account, which bucket,
which credentials — are deliberately **not** here: they are not project knowledge, and this file is
public. What is here is the shape of a run and the invariants, each of which comes from a run that
went wrong.

For *what* still needs measuring, see [`../rented-hardware-test-plan.md`](../rented-hardware-test-plan.md).

## Why EC2 rather than CI or a developer machine

| | limit |
|---|---|
| **CI** | cannot choose a CPU (GitHub mixes runner hardware), has no GPU, cannot hold a 44 GB model, and is not meant to run for an hour |
| **a developer machine** | has exactly one CPU and one GPU — which is the whole problem, since the project ships 22 CPU backend modules |
| **EC2** | one instance family per microarchitecture, on demand, billed per second |

The instance families that map to a CPU module, measured as available in `us-east-1`:

| family | CPU | module it exercises |
|---|---|---|
| `m8i` | Granite Rapids | `sapphirerapids` (AMX) — and every level below it |
| `m7i` / `c7i` | Sapphire Rapids | `sapphirerapids` (AMX) |
| `m7a` / `c7a` | EPYC Genoa (Zen 4) | `zen4` |
| `m6i` / `c6i` | Ice Lake-SP | `icelake` |
| `m5n` / `c5n` | Cascade Lake | `cascadelake` |
| `m5` / `c5` | Skylake-SP | `skylakex` |
| `m6a` / `c6a` | EPYC Milan (Zen 3) | `haswell` |
| `m7g` / `c8g` | Graviton (aarch64) | the `armv*` list |
| `g4dn` / `g5` / `g6` | NVIDIA T4 / A10G / L4 | the `cuda13`, `vulkan`, `opencl` **module jars** |
| `r7i` | Intel, up to 2 TB RAM | a model that does not fit a workstation (the 78B Kolibri-1 run used 123 GB) |

Four x86 levels stay unreachable and can only ever be *load*-tested: `piledriver` (no Bulldozer-era
instance exists), `cannonlake` (barely shipped), `alderlake` (consumer Intel) and `cooperlake`. The
two SME aarch64 modules are unreachable anywhere.

## The shape of a run

No interactive login, no SSH, no RDP. The instance gets a startup script that does everything and
streams its log out after each step:

1. **Dead-man switch first**, before anything else can hang.
2. Install a JDK, fetch one archive holding the classes, the runtime jars and one natives tree, fetch
   the model.
3. Stage and measure, appending to a log and uploading it after every step.
4. Shut down, which terminates.

A watcher on the developer machine polls the log, reports progress, and terminates the instance when
the log says it finished — or when its own timeout expires.

**On the result channel**: the log has to leave the instance by some route the instance can write to
without credentials. A pre-signed upload URL does that (the authorisation is in the URL, so nothing
secret lands on a throwaway machine). The EC2 console log is **not** a usable channel: on Windows
`user-data` output never reaches it at all, and on Linux it lags minutes behind.

## The invariants, each from a run that went wrong

1. **Always terminate, never stop.** A terminated instance costs nothing afterwards; a stopped one
   keeps paying for its disk. Nothing in a measurement needs to survive between runs.
2. **A dead-man switch is the first line of the script** — `shutdown -h +<minutes>` with the instance
   set to terminate on shutdown. Without it, a script that hangs runs until something external kills
   it, and nothing does once the watcher on the developer machine dies. With it, the worst case is a
   number chosen in advance.
3. **Many measurements per boot.** Booting, installing a JDK and fetching a model is 10–15 minutes;
   the measurement itself is often seconds. Fourteen modules in one boot cost the same as one.
4. **One variable, same machine.** This is what makes a table mean anything, and it is also what made
   the AMX crash conclusive: one module staged at a time with everything else fixed left the module
   as the only difference.
5. **A result destination named uniquely per run.** Reusing one made a watcher read the *previous*
   run's finished log, declare success and terminate a running instance mid-measurement. Twice.
6. **Read the error lines, not the exit code.** A probe whose every run exits non-zero looks like the
   thing under test is broken and is usually the harness — a mistyped path, a missing package, an
   unexpanded placeholder.
7. **Match the instance OS to the artifact's build OS.** A module built in CI on Ubuntu 24 (glibc
   2.39) does not load on Amazon Linux 2023 (glibc 2.34). The manylinux\_2\_28 CPU jars (glibc 2.28)
   load on both.
8. **Install what the artifact links against.** The Linux CPU modules need `libgomp.so.1`; an
   Ubuntu-minimal image does not have it, and the failure reads
   `UnsatisfiedLinkError: The native library of backend 'cpu' could not be loaded` — which looks like
   a packaging defect and is a missing package. `readelf -d` on the library (or
   `.github/buildcheck/nativedeps.py`, which parses ELF, PE and Mach-O itself) answers this before a
   boot is spent on it.
9. **Generate scripts without backslashes.** A path written into a generated script through a
   language that treats `\f` and `\b` as escapes becomes a form feed or a backspace, and the
   resulting error (`Could not find or load main class C:\pixpkg\…`) costs a full boot. Build such
   paths from `chr(92)` or with forward slashes.
10. **The harness is the first suspect, not the artifact.** Two measurements this project ran looked
   like product defects and were neither. A Windows classpath built in a POSIX shell had `C:/…`
   rewritten to `C` plus `\Users\…`, so Java received broken entries and the loader correctly
   reported no natives jar on the classpath (`MSYS_NO_PATHCONV=1` fixes it). And a local probe
   declared all five SDK layouts broken because `chmod +x` does not stick on that filesystem, so
   `[ -x ]` answered false for every fixture -- including the one that demonstrably works in CI. In
   both cases the tool under test was telling the truth. Reproduce the *expected* case first; if even
   that fails, the harness is wrong.

## Cost, measured

Per-second billing, and every measurement here was cents:

| run | instance | wall clock | approximate cost |
|---|---|---|---|
| 14 CPU modules staged + the full set | `m8i.xlarge` | ~18 min | $0.07 |
| the same on Zen 4 | `m7a.xlarge` | ~18 min | $0.06 |
| a GPU module jar probe | `g4dn.xlarge` | ~20 min | $0.18 |
| the 78B model, both GGUF dialects, 76 GB downloaded | `r7i.4xlarge` | ~75 min | $1.30 |

A **refused** launch — a quota not yet raised, no capacity for that type — costs nothing, so a retry
loop may poll for hours without consequence.

Two things that cost money without an instance running, and are therefore worth a sweep after every
session: an **EBS volume survives a terminate** unless `DeleteOnTermination` was set, and an
unassociated Elastic IP is billed. Neither shows up in an instance query, and both can sit in a
region nobody looks at — so sweep instances, volumes, snapshots and addresses across **every** region,
not just the one that was used.

A **GPU instance family starts at a zero vCPU quota** on a fresh account. Raising it is a support
request measured in hours to days; the On-Demand and Spot quotas are **independent**, with separate
approvals, so filing both is worth it. Both were granted here within a day, which is what finally
made the GPU findings in [`gpu-module-jars.md`](gpu-module-jars.md) measurable -- and the second GPU
run of this project cost under 0.05 USD, because the probe needed four minutes once it started. Note that a Spot instance can only be terminated, never
stopped — which does not matter for a measurement that terminates anyway.
