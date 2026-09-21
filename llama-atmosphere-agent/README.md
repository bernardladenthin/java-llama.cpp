<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT
-->

# llama-atmosphere-agent — a local JVM coding agent on java-llama.cpp

A minimal, copy-and-run **terminal coding agent** (think Claude Code / OpenCode, reduced to the
essentials) that runs entirely on the JVM and entirely offline:

- **Model:** any GGUF served by java-llama.cpp's OpenAI-compatible HTTP surface — either a server
  you start yourself, or the GGUF loaded **in this process**.
- **Agent:** [Atmosphere](https://github.com/Atmosphere/atmosphere)'s built-in OpenAI-compatible
  runtime (`org.atmosphere:atmosphere-ai`): streaming, the model→tool→model loop, and its
  workspace-confined file tools (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`,
  `delete`, `rename`). Driven **headless** — no Spring Boot, no servlet container, no `@Agent`
  scanning — through `BuiltInAgentRuntime`.
- **Shell:** an opt-in `run_command` tool (`--allow-shell`) so the model can build and test.

This folder is a **standalone Maven project**, deliberately *not* a reactor module and *not*
published: CI builds and tests it against the core of the same checkout; you copy the folder, set
`llama.version`, and run it.

## Quick start

Requirements: JDK 21+ and Maven. No native toolchain: the core jar ships the natives.

**A. Against a server you run yourself** (you keep every llama.cpp flag):

```bash
# 1. start java-llama.cpp's full upstream server (WebUI included) from the fat jar of a release;
#    --jinja enables the model's tool-call template, which tool calling needs
java -jar llama-<version>-jar-with-dependencies.jar -m /models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
     --jinja --port 8080 --api-key sk-local
#    (or upstream llama-server with the same flags — any OpenAI-compatible endpoint works)

# 2. run the agent from this folder
mvn -q compile exec:java -Dllama.version=<version> \
    -Dexec.args="--base-url http://127.0.0.1:8080/v1 --workspace /path/to/project --allow-shell"
```

**B. In-process** (one command, the GGUF is loaded into the agent's JVM and served over a loopback
`OpenAiCompatServer`):

```bash
mvn -q compile exec:java -Dllama.version=<version> \
    -Dexec.args="--model /models/Qwen2.5-7B-Instruct-Q4_K_M.gguf --ngl 99 --workspace /path/to/project"
```

GPU natives: pick the core classifier, e.g. `-Dllama.classifier=cuda13-linux-x86-64` or
`vulkan-windows-x86-64` (the vendor runtime must be installed — see the root README's classifier
table). Without it the default CPU jar (incl. macOS Metal) is used.

Then type a request at the `you>` prompt (`/clear` drops the history, `/exit` quits), or run a single
turn with `--prompt "…"`. Streamed text appears as it is generated; every tool call and its result
are printed as `⚙ read_file {path=…}` / `↳ …` lines.

### Options

| Flag | Meaning | Default |
|---|---|---|
| `--base-url <url>` | OpenAI-compatible base URL of a running server | — |
| `--model <file.gguf>` | load this GGUF in-process instead | — |
| `--ngl <n>` / `--ctx-size <n>` | GPU layers / context size for `--model` | `0` / `8192` |
| `--workspace <dir>` | directory the file tools (and `run_command`) are confined to | cwd |
| `--allow-shell` | register `run_command` | off |
| `--system <text>` | replace the default system prompt | built-in |
| `--prompt <text>`, `-p` | one turn, then exit | interactive |
| `--temperature <t>` / `--max-tokens <n>` | sampling / per-call budget | `0.2` / `2048` |
| `--max-tool-rounds <n>` | tool rounds per turn | `25` |
| `--api-key <key>` / `--model-id <id>` | bearer token / `model` field | `sk-local` / `local-model` |

Exactly one of `--base-url` / `--model` is required. Exit code 0 = turn completed, 1 = the turn
errored, 2 = usage error. Set `-Dorg.slf4j.simpleLogger.defaultLogLevel=debug` to see every request.

Pick a **tool-capable instruct model** (Qwen2.5/Qwen3-Instruct, Llama-3.x-Instruct, Mistral,
Hermes, …). Quality of the loop is the model's: a 1.5B model calls one tool and reads its result, a
7B–32B model does multi-step edit/build/test work.

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
- The Spring Boot `@Agent` + WebSocket/SSE UI variant is untested here; it uses the same runtime and
  the same `LLM_BASE_URL`, so it is expected to work but is not CI-covered.
