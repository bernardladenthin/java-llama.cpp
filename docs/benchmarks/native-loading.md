<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# What loading the natives costs, and where it works at all

Measured **2026-10-12** against the **published** 5.2.0-SNAPSHOT jars (`20261012.035453-12`, llama.cpp
b11558) rather than a build tree — so these numbers are what a consumer resolving from Maven actually
gets.

Two questions that had never been measured: what the extraction directory's reuse is worth on
Windows, and which container base images can load the Linux jar at all.

## The extraction directory on Windows

`LlamaLoader` extracts the platform's library directory into a per-build subdirectory of the temp
directory and reuses it across JVM starts. Four properties are documented in `CLAUDE.md`'s
"Modular natives"; all four are confirmed below.

**Method.** The published `cpu-windows-x86-64` jar plus the classes jar and its runtime dependencies,
`net.ladenthin.llama.Diagnose` as the entry point (it loads the library and exits), an isolated
`-Dnet.ladenthin.llama.tmpdir` so the starting state is controlled, three runs each. Windows 11, an
on-access scanner active.

### The reuse is worth about 0.8 s per JVM start

| | run 1 | run 2 | run 3 |
|---|---:|---:|---:|
| **cold** (directory cleared first) | 4854 ms | 2190 ms | 2159 ms |
| **warm** (directory kept) | 1376 ms | 1345 ms | 1376 ms |

The first cold run is the file cache warming, not the measurement; the representative cold figure is
~2175 ms against ~1365 ms warm, so **extracting 18 files / 39 MB costs ~810 ms** and the reuse branch
removes it. That is per JVM start, and therefore also per surefire fork.

### The key includes the module set, visibly

| classpath | directory |
|---|---|
| library jar only | `jllama-backend-cpu-4fc8af71-11b5000` |
| library jar + `vulkan-windows-x86-64` | `jllama-backend-cpu-4fc8af71-11b5000-mf39ba978` |

Two classpaths with different module sets therefore never share a directory, which is what stops one
JVM's set from being mixed into another's.

### Two JVMs of the same build at once

Both started concurrently from the same classpath, both exit 0, both report `load: ok`.

### A running JVM's files survive another JVM's sweep -- exactly the five it still holds

The sharpest test, because it is the one where a wrong answer corrupts a running process. An
`RpcServer` was left running (it loads the library and listens), its directory's mtime was backdated
by 25 minutes so that `STALE_EXTRACTION_AGE` (10 min) had passed, and a second JVM with a *different*
module set — so the first directory is "not on this classpath" — was started.

**18 files before, 5 after.** The five survivors are precisely the ones the running JVM still had
open:

| survivor | why it was locked |
|---|---|
| `jllama.dll` | the library itself |
| `ggml.dll`, `ggml-base.dll` | pre-loaded by full path via `jllama-extras.txt` |
| `ggml-cpu-haswell.dll` | **the CPU module ggml scored best and kept** |
| `ggml-rpc.dll` | in use by the RPC server |

The 13 that went are exactly the CPU modules ggml opens to score and closes again — no longer locked,
so removable. **The `RpcServer` kept running throughout.** The next start of that build re-extracted
the 13 missing files (back to 18) in 1852 ms, between the warm and cold figures above, as copying 13
of 18 files should be.

So the documented behaviour holds in full: the cache is shared, a live JVM is never damaged by it, and
what a sweep takes is re-created on demand.

## Which base images can load the Linux jar

The published `cpu-linux-x86-64` jar declares, across all 18 libraries (read with the project's own
ELF parser, `buildcheck/nativedeps`):

```
ld-linux-x86-64.so.2  libc.so.6  libdl.so.2  libgcc_s.so.1  libm.so.6
libpthread.so.0  libstdc++.so.6   + the siblings next to it
run path: $ORIGIN (every library)      glibc: GLIBC_2.27 at most
```

**`libgomp.so.1` is gone** — the dependency that made the jar fail on a plain Ubuntu server image
(`UnsatisfiedLinkError: The native library of backend 'cpu' could not be loaded`, which reads like a
packaging defect and was a missing distro package). What remains is what a JVM brings with it.

Verified by loading `net.ladenthin.llama.Diagnose` from the published jars inside three containers on
an EC2 instance:

| image | result |
|---|---|
| `gcr.io/distroless/java21-debian12` | **loads** |
| `eclipse-temurin:21-jre` | **loads** |
| `eclipse-temurin:21-jre-alpine` | **refused**, and correctly: `No native library found for os.name=Linux-Musl, os.arch=x86_64` |

The Alpine row is the one worth keeping. It is **not** a defect: `OSInfo` detects musl, looks for a
`Linux-Musl/x86_64` directory, finds none and says so by name. musl was never a shipped platform; what
is new is that the failure is now a named one rather than a dynamic-linker error. A measurement that
only admits successes cannot show where the boundary is, which is why the image that must fail was
part of the matrix.

## What these runs cannot show

* **One Windows machine**, whose CPU selects `ggml-cpu-haswell`. A machine that selects a different
  module would lock a different file; the mechanism is the same, the survivor list is not.
* **The timings include JVM startup** (~270 ms measured with the natives absent), which is why the
  cold-minus-warm difference is quoted rather than either figure alone.
* **No inference.** `Diagnose` loads the library and exits; throughput is measured elsewhere
  ([`cpu-backend-modules.md`](cpu-backend-modules.md)).
* **The container rows prove loading, not running a model** — the same distinction the GPU page makes:
  a successful load has repeatedly proven less than it looks.
