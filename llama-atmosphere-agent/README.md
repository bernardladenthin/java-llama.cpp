<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT
-->

# llama-atmosphere-agent — a local, general-purpose JVM agent on java-llama.cpp

A minimal, copy-and-run **general-purpose terminal agent** (think Claude Code / OpenCode, reduced to
the essentials): it reads and edits files, and with `--allow-shell` it runs any command on your
machine — `docker`, `git`, build tools, system information. It runs entirely on the JVM and entirely
offline:

- **Model:** any GGUF served by java-llama.cpp's OpenAI-compatible HTTP surface — either a server
  you start yourself, or the GGUF loaded **in this process**.
- **Agent:** [Atmosphere](https://github.com/Atmosphere/atmosphere)'s built-in OpenAI-compatible
  runtime (`org.atmosphere:atmosphere-ai`): streaming, the model→tool→model loop, and its
  workspace-confined file tools (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`,
  `delete`, `rename`). Driven **headless** — no Spring Boot, no servlet container, no `@Agent`
  scanning — through `BuiltInAgentRuntime`.
- **Shell:** an opt-in `run_command` tool (`--allow-shell`) that runs any command line through the
  system shell (`cmd.exe` on Windows, `sh` elsewhere).

This folder is a **standalone Maven project**, deliberately *not* a reactor module and *not*
published: CI builds and tests it against the core of the same checkout; you copy the folder and
run it. Its `pom.xml` pins `llama.version` to the release these instructions describe (**5.2.0**);
pass `-Dllama.version=…` to run against another core, e.g. a `-SNAPSHOT` before a release.

> [!WARNING]
> With `--allow-shell` the model runs **any** command it decides to run, with **your** user's
> rights, without asking — deleting files, pushing to git, stopping containers included. Start it on
> a machine and account you are willing to hand to the model, and point `--workspace` at a copy of
> a project, not your only one. Without the flag it can only use the file tools inside `--workspace`.

## Getting started from scratch

You need **JDK 21+** and **Maven** (`mvn -v` must report Java 21 or newer). No C++ toolchain, no
CMake and no separate llama.cpp install: the core jar from Maven Central ships the native libraries
for Windows, Linux and macOS.

**1. Get this folder.** It is not published as an artifact; clone the repository (or download it
as a ZIP from GitHub) and work in `llama-atmosphere-agent/`. The folder is self-contained — you can
copy it anywhere, `.mvn/jvm.config` included:

```bash
git clone --depth 1 https://github.com/bernardladenthin/java-llama.cpp.git
cd java-llama.cpp/llama-atmosphere-agent
```

**2. Get a model.** Any tool-capable instruct GGUF works; a fast default that fits an 8 GB GPU with a
16k context is Qwen3-4B-Instruct-2507 (2.3 GB):

```bash
curl -L --create-dirs -o models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf
```

**3. Start the agent.** The first run downloads `net.ladenthin:llama` and Atmosphere from Maven
Central; then a `you>` prompt appears (`/clear` drops the history, `/exit` quits).

Linux / macOS:

```bash
mvn -q compile exec:java \
    -Dexec.args="--model models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf --ctx-size 16384 --workspace /path/to/project --allow-shell"
```

Windows (PowerShell — quote the whole `-D` argument, no line continuation with `\`):

```powershell
mvn -q compile exec:java "-Dexec.args=--model models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf --ctx-size 16384 --workspace C:\path\to\project --allow-shell"
```

**4. Optional: use the GPU.** Add a core classifier and offload the layers, e.g.
`-Dllama.classifier=vulkan-windows-x86-64` (or `vulkan-linux-x86-64`; any current GPU driver) or
`cuda13-linux-x86-64` (needs the CUDA 13 toolkit), plus `--ngl 99` inside `-Dexec.args`. macOS uses
Metal with the default jar already. The root README's classifier table lists every backend.

## Quick start

**A. Against a java-llama.cpp server that is already running** (you keep every llama.cpp flag):

1. java-llama.cpp is running, for example started from a release fat jar
   (`--jinja` is required for tool calling: it enables the model's tool-call chat template):

   ```bash
   java -jar llama-5.2.0-jar-with-dependencies.jar -m models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf --jinja --port 8080
   ```

   The fat jars are assets of each [GitHub release](https://github.com/bernardladenthin/java-llama.cpp/releases):
   `llama-5.2.0-jar-with-dependencies.jar` runs on the CPU (and Metal on macOS);
   `llama-5.2.0-all-<os>-<arch>-jar-with-dependencies.jar` (`linux-x86-64`, `linux-aarch64`,
   `windows-x86-64`, `windows-aarch64`) additionally carries every GPU backend for that platform and
   uses the first one whose vendor runtime loads — CUDA, ROCm, SYCL, Vulkan, OpenCL, OpenVINO — falling
   back to the CPU; `-Dnet.ladenthin.llama.backend=vulkan` (or `cpu`) forces one. Upstream
   `llama-server` with the same flags works too; any OpenAI-compatible endpoint does.

2. Start the agent from this folder (`llama-atmosphere-agent/`):

   ```bash
   mvn -q compile exec:java \
       -Dexec.args="--base-url http://127.0.0.1:8080/v1 --workspace /path/to/project --allow-shell"
   ```

A `you>` prompt appears. Type a request; the answer streams as it is generated, and every tool call
and its result are printed as `⚙ read_file {path=…}` / `↳ …` lines. `/clear` drops the history,
`/exit` quits.

A single turn without the REPL:

```bash
mvn -q compile exec:java \
    -Dexec.args="--base-url http://127.0.0.1:8080/v1 --workspace /path/to/project --prompt 'Read the README and summarize it'"
```

If the server was started with `--api-key <key>`, add `--api-key <key>` to the agent's arguments.

**B. In-process** (one command, the GGUF is loaded into the agent's JVM and served over a loopback
`OpenAiCompatServer` — no separately running server):

```bash
mvn -q compile exec:java \
    -Dexec.args="--model models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf --ngl 99 --workspace /path/to/project"
```

**Everything at once — shell access and your own system prompt:**

```bash
mvn -q compile exec:java \
    -Dexec.args="--model models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf --ngl 99 --ctx-size 16384 --workspace /path/to/project --allow-shell --system 'You are a local assistant on this machine with full shell access. run_command executes any command line, including docker, git and build tools. When asked about the system, run a command instead of explaining it. Read a file before you edit it. Answer in the language of the user.'"
```

Then ask, for example, *"which docker images are available?"* or *"build the project and fix the
first compiler error"*. On Windows PowerShell, quote the whole argument instead:
`"-Dexec.args=--model C:\models\… --allow-shell --system '…'"`. Inside `--system '…'` avoid the
apostrophe (write *the user* rather than *user's*): the value is already single-quoted.

GPU natives: pick the core classifier, e.g. `-Dllama.classifier=cuda13-linux-x86-64` or
`vulkan-windows-x86-64` (the vendor runtime must be installed — see the root README's classifier
table). Without it the default CPU jar (incl. macOS Metal) is used. In mode A the classifier is
irrelevant: inference stays in the running server, the agent's JVM loads no model.

### Options

| Flag | Meaning | Default |
|---|---|---|
| `--base-url <url>` | OpenAI-compatible base URL of a running server | — |
| `--model <file.gguf>` | load this GGUF in-process instead | — |
| `--ngl <n>` / `--ctx-size <n>` | GPU layers / context size for `--model` | `0` / `8192` |
| `--log-verbosity <n>` / `--verbose` | llama.cpp log threshold for `--model` (1 errors, 2 warnings, 3 info, 4 trace, 5 debug) / log everything | `2` / off |
| `--workspace <dir>` | directory the file tools are confined to, and where `run_command` starts | cwd |
| `--allow-shell` | register `run_command`: any command line, starting in the workspace | off |
| `--system <text>` | replace the default system prompt | built-in |
| `--prompt <text>`, `-p` | one turn, then exit | interactive |
| `--temperature <t>` / `--max-tokens <n>` | sampling / per-call budget | `0.2` / `2048` |
| `--max-tool-rounds <n>` | tool rounds per turn | `25` |
| `--api-key <key>` / `--model-id <id>` | bearer token / `model` field | `sk-local` / `local-model` |

Exactly one of `--base-url` / `--model` is required. Exit code 0 = turn completed, 1 = the turn
errored, 2 = usage error. Set `-Dorg.slf4j.simpleLogger.defaultLogLevel=debug` to see every request.

**Console output with `--model`.** llama.cpp writes its own log (`slot …`, `srv …`, model loading)
to **stderr**, the same console the streamed answer goes to on stdout, so at llama.cpp's default
threshold (INFO) the per-request timing lines land in the middle of the answer. The agent therefore
loads the in-process model with `--log-verbosity 2` (warnings and errors only); `--log-verbosity 3`
brings the INFO lines back and `--verbose` logs everything. With `--base-url` the server is a separate
process and keeps its own log settings (`-lv` on `llama-server` / `NativeServer`). Two things stay
true whatever the threshold: the agent's own status lines (`Loading …`, `Endpoint …`) also go to
stderr, and `2> llama.log` therefore hides both. On Windows, loading a model switches the console to
UTF-8 (llama.cpp calls `SetConsoleOutputCP(CP_UTF8)`), while the JVM keeps encoding stdout in the
code page it saw at startup, so umlauts and emoji in the answer would turn into `�` / `?`; the
project's `.mvn/jvm.config` pins `-Dstdout.encoding=UTF-8 -Dstderr.encoding=UTF-8` for the `mvn`
JVM so both sides agree.

### The system prompt

Without `--system` the agent uses a built-in **general-purpose** prompt: it names the file tools and
the workspace they work on, and — only with `--allow-shell` — states that `run_command` runs *any*
command line on this machine (the shell is named, so the model writes the right syntax) and that the
model should run a command rather than explain one. Without `--allow-shell` it tells the model it
cannot run commands and to suggest the flag, so the model does not invent a limitation of its own.

The wording is plain text, not Java: [`src/main/resources/net/ladenthin/llama/atmosphere/`](src/main/resources/net/ladenthin/llama/atmosphere/)
holds `system-prompt.txt` (placeholders `{workspace}` and `{shell_section}`), `system-prompt-shell.txt` /
`system-prompt-no-shell.txt` (the `{shell_section}` with and without `--allow-shell`; `{shell}` is the
shell's name) and `run-command-tool.txt` (the `run_command` description the model reads). Edit them
there to change the default for everyone; `--system` overrides it per run.

This wording matters more than it looks: an earlier default called the agent a *coding agent* and
described `run_command` as a way to *"build, test or inspect the project"*, and Qwen3-4B then refused
*"list the docker images"* ("my tools are only for files") although the tool was registered and the
command worked. `--system <text>` replaces the default **completely** — include whatever the model
still needs to know (the workspace, the shell, your language) in your own text.

Pick a **tool-capable instruct model** (Qwen2.5/Qwen3-Instruct, Llama-3.x-Instruct, Mistral,
Hermes, …). Quality of the loop is the model's: a 1.5B model calls one tool and reads its result, a
7B–32B model does multi-step edit/build/test work. Qwen3-4B-Instruct-2507 is a good fast default (fits
an 8 GB GPU with a 16k context); Qwen2.5-Coder-7B, in contrast, wrote the call as a JSON code block
into its answer instead of calling the tool.

## What is verified, and where

| Feature | java-llama.cpp | Atmosphere needs it | Test | Change needed |
|---|---|---|---|---|
| `POST /v1/chat/completions` (always `stream:true`) | yes | yes | wire + model | none |
| SSE streaming, `data:`/`[DONE]`, chunk-by-chunk `delta.content` | yes | yes | wire + model | none |
| `system` / `user` / `assistant` / `tool` roles, history replay | yes | yes | wire + model | none |
| `tools` (JSON-Schema function definitions) | forwarded verbatim | yes | wire + model | none |
| streamed `delta.tool_calls` with `index`, `id`, `function.name`, fragmented `arguments` | yes (upstream chunk shape) | yes | wire | none |
| `finish_reason:"tool_calls"` ends the round | yes | yes | wire + model | none |
| assistant `tool_calls` message **without** `content` on the follow-up | accepted (`common_chat_msgs_parse_oaicompat`: `content` *or* `tool_calls`) | sent that way | wire | none |
| `role:"tool"` + `tool_call_id` (+ `name`) | forwarded verbatim | yes | wire + model | none |
| several tool calls in one turn (index 0/1, interleaved fragments) | yes | yes | wire | none |
| 3+ consecutive tool rounds, full history kept | yes | yes | wire (4 rounds) + model (read/write/read) | none |
| `temperature`, `max_tokens` | mapped to `temperature` / `n_predict` | sent | wire | none |
| `tool_choice`, `parallel_tool_calls`, `response_format`, `/v1/responses`, `/v1/embeddings` | available | **not used** by the built-in runtime (Responses API only for `api.openai.com`) | — | none |
| `GET /v1/models` | yes | optional (best-effort enumeration) | wire | none |
| API key | `--api-key` → `401` on mismatch | sends `Authorization: Bearer`; a dummy key is fine | wire | none |
| custom base URL | — | `LLM_BASE_URL` / `AiConfig.configure(...)` | wire + model | none |
| error before the stream starts (401, 413, 500) | HTTP status + JSON error | surfaced as `session.error` (5xx retried) | wire | none |
| **engine error after the stream started** | HTTP 200 already sent → `data: {"error":…}`, no `[DONE]` | **ignored**: the SSE parser reads only `choices[0]`, the turn completes with the text so far | wire (pinned as a known gap) | SHOULD, in Atmosphere's `OpenAiCompatibleClient` |

*wire* = `AtmosphereWireContractTest` / `LocalAgentTest`: the **real** `OpenAiCompatServer` (routing,
bearer auth, `/v1/models`, SSE framing) over a loopback socket with a scripted engine replaying
llama.cpp-shaped chunks; no native library, no model, seconds, on every PR. *model* =
`AtmosphereToolLoopIntegrationTest`: the same loop against the Qwen2.5-1.5B-Instruct tool model in
CI (plain chat, streaming, a tool call whose result is answered, a read→write→read loop that
changes a temp file). It self-skips without the GGUF:

```bash
mvn -f llama-atmosphere-agent/pom.xml test -Dtest=AtmosphereToolLoopIntegrationTest \
    -Dnet.ladenthin.llama.tool.model=models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf -Dnet.ladenthin.llama.test.ngl=0
```

**Verdict: A — Atmosphere's built-in runtime runs a complete streaming tool-calling agent loop
against java-llama.cpp unchanged.** The only change on the java-llama.cpp side was to make the
existing test seam (`OpenAiBackend`, `ChunkSink`, the backend constructor of `OpenAiCompatServer`)
public so this project can drive the real server without a model.

## How the wiring works (all of it)

```java
AiConfig.LlmSettings settings = AiConfig.configure("local", modelId, apiKey, baseUrl); // LLM_MODE/LLM_MODEL/LLM_API_KEY/LLM_BASE_URL
BuiltInAgentRuntime runtime = new BuiltInAgentRuntime();
runtime.configure(settings);

List<ToolDefinition> tools = new ArrayList<>(FileSystemTools.all());        // Atmosphere's file tools
tools.add(ShellTool.definition(workspace, Duration.ofSeconds(120), 20_000)); // optional

AgentExecutionContext context = new AgentExecutionContext(message, systemPrompt, modelId, null, "console",
        null, null, tools, null, null, List.of(), Map.of(), history, null, null);
context = ToolLoopPolicies.attach(context, ToolLoopPolicy.maxIterations(25));
runtime.execute(context, session); // session.injectables() carries the AgentFileSystem the file tools resolve
```

`AgentRunner` is exactly that; `ConsoleSession` renders the stream and supplies the
`WorkspaceAgentFileSystem` (path-confined, size-limited) through `injectables()`; `LocalAgent` parses
the options and optionally hosts the model. The `@Agent`/`@AiTool` annotations and the Spring Boot
starter are the *deployment* layer on top of the same runtime — not needed for a local terminal agent.

## Limitations / next steps

- Tool rounds are not kept in the cross-turn history (only `user`/`assistant` text is replayed).
- No approval prompts for destructive tools yet (`ToolDefinition.requiresApproval` exists in Atmosphere).
- An engine error after the stream started ends the turn silently (see the table).
- **One in-process agent per machine at a time.** The core extracts its native library to a fixed
  name (`jllama.dll` / `libjllama.so` in the temp directory); on Windows a second JVM cannot replace
  the file while the first one has it loaded, so a second `--model` agent fails at startup with
  `Failed to delete old native lib` followed by `No native library found`. Two in-process models
  also share the GPU's memory — a second 4B model on an 8 GB card can stall instead of failing.
  To run several agents, start one server (mode A) and point each agent at it with `--base-url`.
- The Spring Boot `@Agent` + WebSocket/SSE UI variant is untested here; it uses the same runtime and
  the same `LLM_BASE_URL`, so it is expected to work but is not CI-covered.
