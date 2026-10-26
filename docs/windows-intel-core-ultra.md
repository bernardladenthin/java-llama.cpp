<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT
-->

# Windows on an Intel Core Ultra: which artifacts, and how to start them

For a Windows machine that has **Maven and several JDKs in `C:\Program Files\Java`** and nothing
else — no JBang, no C++ toolchain, no checkout required. Worked through on an **Intel Core Ultra 7
165U** (Meteor Lake-U), and valid for any Windows x86-64 machine with an Intel iGPU; only the
instruction-set paragraph differs between CPU generations.

Everything runs out of the local Maven repository: one `go-offline` up front, then `mvn exec:java`
with the JDK you choose per window. For the full classifier table and the general rules, see the
README's [Choosing the natives jars](../README.md#choosing-the-natives-jars).

## Which artifacts

| Take | Size | Why |
|---|--:|---|
| `net.ladenthin:llama:6.0.0` | 0.38 MiB | the Java classes |
| `net.ladenthin:llama:6.0.0:cpu-windows-x86-64` | 16.62 MiB | `jllama.dll` + llama/ggml + all 14 CPU backend modules |
| `net.ladenthin:llama:6.0.0:vulkan-windows-x86-64` | 12.01 MiB | the Arc iGPU, through the installed driver's `vulkan-1.dll` — nothing else to install |

That is the whole runtime, 29 MiB. What the hardware implies:

- **AVX2 + AVX-VNNI, no AVX-512, no AMX** (Core Ultra series 1 ships AVX-512 disabled) → ggml loads
  **`ggml-cpu-alderlake.dll`** out of the library jar. That variant exists only because the shipped
  Windows CPU jar is built with plain clang; `clang-cl` cannot build it at all.
- **Arc Graphics (Xe-LPG, 4 Xe-cores) sharing LPDDR5 with the CPU** → Vulkan helps prompt
  processing; token generation is memory-bandwidth-bound and both engines pull from the same RAM, at
  a 15 W package budget. Measure both directions before believing either.
- **2 P-cores + 8 E-cores + 2 LP-E-cores** → do not use all 14 threads, see [Thread
  count](#thread-count).

**Skip** `sycl-windows-x86-64` unless you want to install the oneAPI 2026 runtime (`sycl9.dll`,
`mkl_sycl_blas.6.dll`, `dnnl.dll`, `tbb12.dll`, `ze_loader.dll`) on every machine;
`opencl-windows-x86-64` (ggml's OpenCL backend is tuned for Qualcomm Adreno); `cuda13-*` and
`rocm-*` (no NVIDIA, no AMD GPU); and every non-Windows-x86-64 jar. `openvino-windows-x86-64` is the
only path to the **NPU** and is a special case — see [the last section](#the-npu). **Do not use
`llama-platform`**: it carries the library jars of all six desktop platforms (~130 MiB) for the one
you need.

## Two JDKs, and what needs which

| | JDK floor |
|---|---|
| the core, both servers, the natives | **Java 8+** |
| `llama-atmosphere-agent` | **Java 21+** (Atmosphere's own floor) |

The agent is Java 21 bytecode; on a JDK 8 it fails in the compiler with `invalid target release: 21`
and no flag helps. So the server runs on the 8, the agent on the 21, and `JAVA_HOME` decides which —
per window, because `exec:java` runs in Maven's own JVM:

```powershell
# the window that starts the agent
$env:JAVA_HOME = 'C:\Program Files\Java\openjdk21.0.11'
mvn -v    # must report: Java version: 21...
```

Leave the machine-wide `JAVA_HOME` on the JDK 8 so the server keeps working in every other window.
Nothing else needs setting: Maven resolves the classpath from `~\.m2\repository` itself.

## Step 1 — two poms

Two directories, two small poms. They are separate because the two halves need different JDKs and
different dependencies.

**`C:\llama\server\pom.xml`:**

```xml
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    <groupId>local</groupId>
    <artifactId>llama-server-windows</artifactId>
    <version>1</version>
    <packaging>pom</packaging>
    <properties>
        <llama.version>6.0.0</llama.version>
        <!-- a property, not the plugin's <mainClass>: that way -Dexec.mainClass=... on the
             command line can override it (an explicit <mainClass> would win over the CLI) -->
        <exec.mainClass>net.ladenthin.llama.server.ServerLauncher</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>net.ladenthin</groupId>
            <artifactId>llama</artifactId>
            <version>${llama.version}</version>
        </dependency>
        <dependency>
            <groupId>net.ladenthin</groupId>
            <artifactId>llama</artifactId>
            <version>${llama.version}</version>
            <classifier>cpu-windows-x86-64</classifier>
        </dependency>
        <dependency>
            <groupId>net.ladenthin</groupId>
            <artifactId>llama</artifactId>
            <version>${llama.version}</version>
            <classifier>vulkan-windows-x86-64</classifier>
        </dependency>
    </dependencies>
    <build>
        <plugins>
            <plugin>
                <groupId>org.codehaus.mojo</groupId>
                <artifactId>exec-maven-plugin</artifactId>
                <version>3.6.4</version>
                <configuration>
                    <!-- a server never returns; let it own the JVM -->
                    <cleanupDaemonThreads>false</cleanupDaemonThreads>
                </configuration>
            </plugin>
        </plugins>
    </build>
</project>
```

**`C:\llama\agent\pom.xml`** — same shape, the agent instead of the server. `llama-platform` is
excluded because the agent's published pom pulls the library jars of all six desktop platforms:

```xml
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    <groupId>local</groupId>
    <artifactId>llama-agent-windows</artifactId>
    <version>1</version>
    <packaging>pom</packaging>
    <properties>
        <llama.version>6.0.0</llama.version>
        <exec.mainClass>net.ladenthin.llama.atmosphere.LocalAgent</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>net.ladenthin</groupId>
            <artifactId>llama-atmosphere-agent</artifactId>
            <version>${llama.version}</version>
            <exclusions>
                <exclusion>
                    <groupId>net.ladenthin</groupId>
                    <artifactId>llama-platform</artifactId>
                </exclusion>
            </exclusions>
        </dependency>
        <!-- only needed for --model (the agent loading a GGUF itself);
             with --base-url against the server it loads no natives at all -->
        <dependency>
            <groupId>net.ladenthin</groupId>
            <artifactId>llama</artifactId>
            <version>${llama.version}</version>
            <classifier>cpu-windows-x86-64</classifier>
        </dependency>
        <dependency>
            <groupId>net.ladenthin</groupId>
            <artifactId>llama</artifactId>
            <version>${llama.version}</version>
            <classifier>vulkan-windows-x86-64</classifier>
        </dependency>
    </dependencies>
    <build>
        <plugins>
            <plugin>
                <groupId>org.codehaus.mojo</groupId>
                <artifactId>exec-maven-plugin</artifactId>
                <version>3.6.4</version>
                <configuration>
                    <cleanupDaemonThreads>false</cleanupDaemonThreads>
                </configuration>
            </plugin>
        </plugins>
    </build>
</project>
```

## Step 2 — pre-load everything, once

Once per pom, with network. This downloads every artifact, every transitive dependency **and the
plugins** into `~\.m2\repository`:

```powershell
cd C:\llama\server ; mvn dependency:go-offline
cd C:\llama\agent  ; mvn dependency:go-offline
```

Either JDK does this — Maven only downloads here, it neither compiles nor loads the jars, so the
JDK 8 can fetch the agent too. Afterwards every command below runs with `-o` and touches no network.

Two measured notes: the plugin cache is why this is per pom rather than once globally (without it the
first offline run fails with `No plugin found for prefix 'exec'`), and `go-offline` resolves the tree
**without honouring `<exclusions>`**, so the agent pom also caches the other five platforms' library
jars (~113 MiB). For a pre-load that is simply "have everything local", and it changes nothing about
what runs — the effective classpath still honours the exclusion.

## Step 3 — the server on the JDK 8, with a port to test

Put the GGUF anywhere, e.g. `C:\models\`. Any tool-capable instruct model works; a reasonable
default here is Qwen3-4B-Instruct-2507-Q4_K_M (2.3 GB).

```powershell
cd C:\llama\server
mvn -o exec:java "-Dexec.args=-m C:\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf --host 127.0.0.1 --port 9931 -c 16384 -ngl 99 -t 8 -tb 10 --jinja"
```

This is the **real upstream `llama_server` running inside `libjllama`** over JNI, so every
llama-server flag works verbatim and the bundled WebUI is served. `--jinja` enables the model's chat
template, which is what makes tool calling work — the agent in step 4 needs it. `-ngl 99` offloads
to the iGPU through Vulkan; drop it to compare against the CPU.

Test the port from another window:

```powershell
Invoke-RestMethod http://127.0.0.1:9931/health
Invoke-RestMethod -Method Post http://127.0.0.1:9931/v1/chat/completions -ContentType application/json -Body '{"messages":[{"role":"user","content":"2+2=?"}],"max_tokens":16}'
```

and open <http://127.0.0.1:9931/> in a browser for the WebUI. The OpenAI-compatible API lives on the
same port under `/v1/…`. (To embed the server in your own Java 8 code instead, see
[`NativeServer`](../README.md#native-server-with-the-built-in-webui-nativeserver).)

## Step 4 — the Atmosphere agent on the JDK 21

Leave the server running. In a new window, point `JAVA_HOME` at the 21 and start the agent against
that server — the model stays loaded once, in the server, and the agent drives it over HTTP:

```powershell
$env:JAVA_HOME = 'C:\Program Files\Java\openjdk21.0.11'
cd C:\llama\agent
mvn -o -q exec:java "-Dexec.args=--base-url http://127.0.0.1:9931/v1 --workspace C:\path\to\project --allow-shell"
```

A `you>` prompt appears. `/clear` drops the history, `/exit` quits, `--prompt "…"` runs a single turn
and exits, `--plain` switches to a line-oriented console for piping or logging.

For **Atmosphere's web console on its own port** instead of the terminal, add `--web`; it listens on
`127.0.0.1:8787` and prints a URL with a fresh access token on every start:

```powershell
mvn -o -q exec:java "-Dexec.args=--base-url http://127.0.0.1:9931/v1 --workspace C:\path\to\project --allow-shell --web --web-port 8787 --web-token my-own-access-token-123"
```

`--web-port 0` takes any free port, and `--web-host 0.0.0.0` exists but warns loudly — tunnel over
SSH rather than binding to the network. One conversation is served: every tab holding the token
shares it. For an IDE, `--acp` speaks the [Agent Client
Protocol](https://agentclientprotocol.com) on stdin/stdout instead; `--web`, `--acp` and `--prompt`
exclude each other.

Without a running server the agent can load the GGUF itself (`--model` instead of `--base-url`),
which is what the two natives dependencies in its pom are for:

```powershell
mvn -o -q exec:java "-Dexec.args=--model C:\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf --ngl 99 --ctx-size 16384 --workspace C:\path\to\project --allow-shell"
```

> [!WARNING]
> `--allow-shell` lets the model run **any** command with your rights. It asks per call by default;
> `--auto` (and answering `[a]`) turns that off for the session. Point `--workspace` at a copy of a
> project, not your only one.

## Thread count

The biggest single knob on this CPU. ggml runs its own `std::thread` pool on Windows (OpenMP is
deliberately off — measured ~2× faster token generation) and synchronises on a barrier per
operation. **The barrier waits for the slowest thread**, and the two LP-E cores in the SoC tile are
considerably slower than the P- and E-cores, so all 14 threads is very likely not the optimum.

Measure `-t 6`, `-t 8` and `-t 10` and take the best. `-tb` sets the prompt-processing thread count
separately and tolerates a higher value, since prompt processing has enough work per barrier to hide
the synchronisation cost.

## Verifying what loaded

The start-up log names the library, the modules put in place and the devices ggml ended up with:

```
load_backend: loaded Vulkan backend from <dir>\ggml-vulkan.dll
load_backend: loaded CPU backend from <dir>\ggml-cpu-alderlake.dll
load_backends_next_to_this_library: 3 backend(s) registered [RPC, Vulkan, CPU], 2 device(s): Vulkan0 (Vulkan), CPU (CPU)
[jllama] native backend 'cpu' loaded from <dir> with GPU module(s) vulkan; 18 file(s) in place for ggml to load
```

The `ggml-cpu-<variant>.dll` line is the one that tells you whether this CPU got `alderlake`. A
one-line build stamp per natives directory (llama.cpp tag, commit, patches, compiler) comes from:

```powershell
cd C:\llama\server
mvn -o exec:java -Dexec.mainClass=net.ladenthin.llama.Diagnose
```

(Measured: that override works only because the poms above declare the main class as the **property**
`exec.mainClass`. Put it in the plugin's `<configuration><mainClass>` instead and the pom wins over
the command line — `-Dexec.mainClass=…Diagnose` then silently starts the server.)

`-Dnet.ladenthin.llama.backend=<a>,<b>` is a **filter** over the module jars on the classpath, not a
switch that adds anything: `cpu` puts no module in place (the clean CPU baseline for a comparison),
`vulkan` only that one. A name that is not a module backend, or whose jar is not on the classpath,
fails the load with a message naming the jar to add.

## The NPU

`openvino-windows-x86-64` is the only module that can address an Intel **NPU** or iGPU through
OpenVINO, and it is **opt-in** by design: its jar on the classpath does nothing until
`net.ladenthin.llama.backend` names it. The reason is measured, in
[`benchmarks/gpu-module-jars.md`](benchmarks/gpu-module-jars.md): OpenVINO's **CPU plugin** opens
successfully and contributes a *device*, which llama.cpp then prefers — on a host without an Intel
accelerator that dropped throughput from 490.4/34.5 to 29.4/**0.2** tokens per second. Every other
module jar is additive because it cannot do this; OpenVINO is the exception the opt-in list exists
for.

This machine does have Intel accelerators, so OpenVINO may well pick one — but that is **not
measured here**, the device choice is OpenVINO's own, and it needs the Intel OpenVINO runtime DLLs
on `PATH`. To try it: add a fourth dependency with classifier `openvino-windows-x86-64` to the server
pom, re-run `go-offline`, and name both modules explicitly.

```powershell
mvn -o exec:java "-Dexec.args=-m C:\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf --port 9931 -c 16384 -ngl 99 -lv 4" "-Dnet.ladenthin.llama.backend=vulkan,openvino"
```

Read the device list in the log before trusting any number: if a device named like OpenVINO's CPU
plugin appears and is used, you are measuring the slow path, not the NPU.
