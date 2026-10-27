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
set per window, because `exec:java` runs in Maven's own JVM. Both steps below open with exactly these
two lines, naming the JDK that step needs:

```bat
set "JAVA_HOME=C:\Program Files\Java\openjdk1.8.0_491"
set "PATH=%JAVA_HOME%\bin;%PATH%"
mvn -v
```

`mvn -v` must then report that JDK (`Java version: 1.8.0_491`, or `21.0.11` for the agent) — that
number decides, not `java -version`. The two `set` lines live and die with the window, so the
machine-wide `JAVA_HOME` can stay wherever it is. Nothing else needs setting: Maven resolves the
classpath from `%USERPROFILE%\.m2\repository` itself.

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
        <!-- needed only when the agent loads a GGUF itself (its "model" option);
             driving a running server instead, it loads no natives at all -->
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
plugins** into `%USERPROFILE%\.m2\repository`:

```bat
cd /d C:\llama\server
mvn dependency:go-offline

cd /d C:\llama\agent
mvn dependency:go-offline
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

```bat
set "JAVA_HOME=C:\Program Files\Java\openjdk1.8.0_491"
set "PATH=%JAVA_HOME%\bin;%PATH%"

cd /d C:\llama\server
mvn -o exec:java -Dexec.args="-m C:\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf --host 127.0.0.1 --port 9931 -c 16384 -ngl 99 -t 8 -tb 10 --jinja"
```

This is the **real upstream `llama_server` running inside `libjllama`** over JNI, so every
llama-server flag works verbatim and the bundled WebUI is served. `--jinja` enables the model's chat
template, which is what makes tool calling work — the agent in step 4 needs it. `-ngl 99` offloads
to the iGPU through Vulkan; drop it to compare against the CPU.

Test the port from another window (`curl.exe`, not `curl` — in PowerShell 5.1 `curl` is an alias for
`Invoke-WebRequest`):

```bat
curl.exe http://127.0.0.1:9931/health

curl.exe http://127.0.0.1:9931/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"2+2=?\"}],\"max_tokens\":16}"
```

and open <http://127.0.0.1:9931/> in a browser for the WebUI. The OpenAI-compatible API lives on the
same port under `/v1/…`. (To embed the server in your own Java 8 code instead, see
[`NativeServer`](../README.md#native-server-with-the-built-in-webui-nativeserver).)

## Step 4 — the Atmosphere agent on the JDK 21

Leave the server running. In a **new** window — this is where the other JDK goes — start the agent
against that server, so the model stays loaded once, in the server, and the agent drives it over
HTTP:

```bat
set "JAVA_HOME=C:\Program Files\Java\openjdk21.0.11"
set "PATH=%JAVA_HOME%\bin;%PATH%"

cd /d C:\llama\agent
mvn -o -q exec:java -Dexec.args="--base-url http://127.0.0.1:9931/v1 --workspace C:\path\to\project --allow-shell"
```

A `you>` prompt appears. `/clear` drops the history, `/exit` quits, `--prompt "…"` runs a single turn
and exits, `--plain` switches to a line-oriented console for piping or logging.

> [!IMPORTANT]
> **The agent's flags are not the server's.** The server takes llama.cpp's argv verbatim (`-m`, `-c`,
> `-ngl`), the agent has its own parser and accepts only the long forms — `--model`, `--ctx-size`,
> `--ngl`/`--gpu-layers`. Passing the server's `-m` to the agent fails with `unknown argument: -m`
> followed by its usage block. Flags it does not know are never ignored, which is deliberate.

**Atmosphere's web console** is `--web`. It is not an addition to the terminal but a replacement for
it: `--web`, `--acp` and `--prompt` exclude each other, so one agent process serves exactly one
front end. To have both, start a **second** agent process against the same server — the model stays
loaded once, in the server, and each agent is only an HTTP client of it. A third window, the JDK 21
again:

```bat
set "JAVA_HOME=C:\Program Files\Java\openjdk21.0.11"
set "PATH=%JAVA_HOME%\bin;%PATH%"

cd /d C:\llama\agent
mvn -o -q exec:java -Dexec.args="--base-url http://127.0.0.1:9931/v1 --workspace C:\path\to\project --allow-shell --web --web-port 8787"
```

It prints `Open in a browser: http://127.0.0.1:8787/?token=<token>`. **The token is part of the
URL** and is a new random one on every start, unless `--web-token` pins it (16 or more characters of
`A-Z a-z 0-9 _ -`) — which is what makes the address worth bookmarking. `--web-port 0` takes any
free port. Binding is loopback only, and `--web-host 0.0.0.0` exists but warns loudly: tunnel with
`ssh -L 8787:127.0.0.1:8787 <user>@<this-host>` and open the same address on the other machine
instead. One conversation per process — every tab holding the token shares it, and a message sent
while another is running replaces it, as typing over a running turn does on the console.
`/mode auto` in the browser stops the approval prompts for the session. The console agent in the
other window keeps its **own** history: the two share the model, not the conversation.

The console's pages come from `atmosphere-spring-boot-starter`, which the agent pom carries for
those resources alone (every transitive Spring dependency excluded), so Step 2 already cached them
and `--web` works offline. If they were missing, the agent says so on stderr and still serves the
endpoint.

> [!NOTE]
> The server of Step 3 runs with one slot, so two agents **serialise**: the second waits for the
> first to finish its turn. `--parallel 2` on the server gives each its own slot, and that splits
> `-c 16384` into 8192 per slot — so it is not free. (`--kv-unified` makes the slots share one pool
> at the full size instead, which trades the guarantee for the capacity.)

For an IDE, `--acp` speaks the [Agent Client Protocol](https://agentclientprotocol.com) on
stdin/stdout instead.

Without a running server the agent can load the GGUF itself (`--model` instead of `--base-url`),
which is what the two natives dependencies in its pom are for:

```bat
mvn -o -q exec:java -Dexec.args="--model C:\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf --ngl 99 --ctx-size 16384 --workspace C:\path\to\project --allow-shell"
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

## gpt-oss-20b

ggml-org's `gpt-oss-20b-mxfp4.gguf` (12.1 GB: the experts in MXFP4, everything else Q8_0) runs on
the CPU modules. It needs about 13 GB of RAM plus the KV cache, so a 32 GB machine holds it next to
Windows. The notes below were **measured in a Linux sandbox with the `alderlake` module staged alone,
not on this laptop**. [`benchmarks/moe-mxfp4-cpu-alderlake.md`](benchmarks/moe-mxfp4-cpu-alderlake.md)
has the numbers and their limits:

- **Leave the KV cache at f16**, i.e. no `-ctk`/`-ctv`. With `-ctk q8_0 -ctv q8_0`, prompt processing
  at an 8k context fell to a quarter (30.8 → 7.5 t/s): ggml's fast flash-attention paths need an f16
  cache with K and V of the same type. For an agent, which re-sends the whole conversation every
  turn, that is the number that counts. The f16 cache costs 1.6 GB at `-c 65536`.
- **Repack and flash attention stay at their defaults**, which were the fastest.
- **Measure `-tb` for this model separately.** Upstream llama.cpp (and the 6.0.0 jar) splits the
  expert matmuls statically across threads, in prompt processing as well, so every thread waits for
  the slowest one. With one core at half speed, prompt processing halved. The LP-E cores are exactly
  such cores, so the `-tb` that suits a dense model may be too high here.

```bat
mvn -o exec:java -Dexec.args="-m C:\models\gpt-oss-20b-mxfp4.gguf --host 127.0.0.1 --port 9931 -c 32768 -ngl 0 -t 8 -tb 8 --jinja"
```

`-ngl 0` keeps it on the CPU modules, the configuration measured. Through Vulkan on the Arc iGPU
(`-ngl 99`) it is unmeasured for this model.

### Both expert paths, measured on this machine

From the release after 6.0.0 on, the natives carry `llama/patches/0022` (6.0.0 itself runs upstream's
path only): the tokens routed to one expert are multiplied
together and the work goes to the threads in small units, so the E- and LP-E-cores take less of it.
In the sandbox that gave +47 % prompt processing and +26 % token generation. Upstream's path stays
switchable, so the difference can be measured here, where the cores really differ:
`LlamaModel.setCpuMoeBatched(false)` in your own code, or for the server the environment variable,
set in the window **before** `mvn` starts:

```bat
:: upstream's path; leave the line out (or open a new window) for the batched default
set GGML_CPU_REPACK_MMID_BATCHED=0
mvn -o exec:java -Dexec.args="-m C:\models\gpt-oss-20b-mxfp4.gguf --port 9931 -c 32768 -ngl 0 -t 8 -tb 8 --jinja"
```

Then send the same long prompt to each run and compare the `timings` of the response, e.g.
`prompt_per_second` and `predicted_per_second` from
`curl.exe http://127.0.0.1:9931/completion -d "{\"prompt\":\"...\",\"n_predict\":128,\"cache_prompt\":false}"`.
A second request with the same prompt and `cache_prompt` on would take the prompt from the cache and
measure nothing.

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

```bat
cd /d C:\llama\server
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

```bat
mvn -o exec:java -Dnet.ladenthin.llama.backend=vulkan,openvino -Dexec.args="-m C:\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf --port 9931 -c 16384 -ngl 99 -lv 4"
```

Read the device list in the log before trusting any number: if a device named like OpenVINO's CPU
plugin appears and is used, you are measuring the slow path, not the NPU.
