<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT
-->

# Feature Investigation — IDE coding/agent backend (2025–2026)

> **Implementation status (this repo).** The XS/S/M recommendations below are implemented on
> `net.ladenthin.llama.server.OpenAiCompatServer`: `POST /infill` (FIM autocomplete), `POST /v1/rerank`
> (RAG), `stream_options.include_usage` passthrough + a `cached_tokens` safety net, `response_format`
> (structured outputs), CORS/`OPTIONS` preflight, bare-path (`/v1`-less) aliases, a `cache_prompt=true`
> default, and `--mmproj`/`--embedding`/`--reranking` CLI flags. Agentic tool-calling is the primary
> target and is verified wire-correct by a C++ guard pinning `tool_calls.function.arguments` as a JSON
> string (llama.cpp #20198). Open items that need a product decision (Ollama native-API emulation,
> Anthropic `POST /v1/messages` + OpenAI `POST /v1/responses` shims, Continue's native `/completion`,
> a per-model FIM template registry, `/props` capability reporting) are tracked in
> [`../TODO.md`](../TODO.md). The verbatim deep-research report follows.

---

# Making a llama.cpp-Backed Local Server a First-Class IDE Coding/Agent Backend (2025–2026)

## TL;DR
- **The single highest-leverage change is to add a llama.cpp-native `POST /infill` endpoint (fields `input_prefix`, `input_suffix`, `input_extra`, `prompt`, `n_predict`), because every high-quality local ghost-text client (llama.vscode, llama.vim, Tabby, Twinny, and Continue's `llama.cpp` provider) drives FIM through `/infill` or a raw `/v1/completions` `suffix` template — NOT through `/v1/chat/completions`.** A chat-only server unlocks chat/agent but currently unlocks zero first-class autocomplete.
- **For chat + agent (Copilot BYOK, Cline, Roo Code, Continue, ProxyAI, Zed, Aider), your existing `/v1/chat/completions` is already the right surface — but the make-or-break details are: streamed `delta.tool_calls` with correct `index`/`id`/`function.name`/`function.arguments` fragments, `finish_reason:"tool_calls"` on the terminating chunk, a `stream_options.include_usage` final usage chunk with an empty `choices` array, and never emitting `tool_calls.function.arguments` as a JSON object (it must be a JSON-encoded string).** Copilot's VS Code custom-endpoint feature also reads `usage.prompt_tokens_details.cached_tokens` and crashes if it is absent.
- **As of VS Code 1.122 (released May 28, 2026), the generic OpenAI-compatible "Custom Endpoint" provider (apiTypes `chat-completions` / `responses` / `messages`) is now in VS Code Stable — so a plain OpenAI-compatible server is now a first-class Copilot chat/agent backend without Insiders or Ollama emulation.** Copilot inline completions, however, remain closed to all local endpoints ("Inline suggestions and next edit suggestions still require a GitHub sign-in. BYOK powers chat, tools, and MCP servers only").

## Key Findings

1. **Two protocol families, not one.** Autocomplete/FIM and chat/agent are almost entirely disjoint wire contracts. Chat/agent is OpenAI `/v1/chat/completions` (or Anthropic `/v1/messages`, or OpenAI `/v1/responses`). Autocomplete is either llama.cpp `/infill`, Ollama `/api/generate`, or raw `/v1/completions` with a `suffix` field and a model-specific FIM template. Your server implements the chat side well and the FIM side not at all.

2. **Copilot inline completion is closed to local models.** Per the VS Code 1.122 release notes, "Inline suggestions and next edit suggestions (NES) still require a GitHub sign-in. BYOK powers chat, tools, and MCP servers only." VS Code's language-models docs add: "Currently, you cannot connect to a local model for inline suggestions. VS Code provides an extension API `InlineCompletionItemProvider` that enables extensions to contribute a custom completion provider." So no llama.cpp server can power Copilot ghost text — you can only target Copilot's **chat + agent** surfaces (or ship your own inline-completion VS Code extension).

3. **Copilot's OpenAI-compatible path went Stable in May 2026.** VS Code 1.122 (May 28, 2026) notes: "The Custom Endpoint provider lets you connect models that implement Chat Completions, Responses, or Messages APIs… The Custom Endpoint provider is now available in VS Code Stable." This supersedes the earlier Insiders-only status and reduces the urgency of emulating Ollama's native API. The built-in **Ollama** provider (native `/api/version`, `/api/tags`, `/api/show`) and the deprecated `github.copilot.chat.customOAIModels` settings object remain as alternative paths. BYOK "now works without GitHub sign-in… in air-gapped or restricted environments" (GitHub Changelog, Apr 22, 2026), though model selection in the UI generally still prompts a GitHub login.

4. **Cline and Roo Code diverge on tool-calling.** Roo Code forces native OpenAI tool calling: per the Roo Code blog ("Sorry we didn't listen sooner: Native Tool Calling"), "In 3.36.0 we introduced native tool calling… In 3.37.0 we made native tool calling the default and removed XML tool calling entirely." If your endpoint doesn't fully implement `tools`/`tool_calls`, Roo (≥3.37) cannot be used unless the user rolls back to 3.36.16 and selects XML in advanced settings. Cline historically inlines XML-style tool instructions into the prompt and parses tool calls out of plain text, so it is tolerant of weak native tool support. This is a critical compatibility fork.

5. **Real-world SSE bugs cluster around three things:** the trailing usage chunk (`stream_options.include_usage`), the `finish_reason` after streamed tool calls (must be `"tool_calls"`, not `"stop"`), and Copilot's hard dependency on `usage.prompt_tokens_details.cached_tokens`.

6. **KV-cache prefix reuse is a latency feature clients actively rely on.** llama.vscode warms the server with a fire-and-forget `/infill` `n_predict:0` request [DeepWiki](https://deepwiki.com/ggml-org/llama.vscode/3.2-llamaserver) and sets `cache_prompt:true`; [GitHub](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) `--cache-reuse 256` is a standard launch flag. For acceptable repeated-prefix latency you must support `cache_prompt`/prompt-prefix reuse.

## Details

### A. Landscape of clients targeting local endpoints

| Client | IDEs | License | Local-endpoint mechanism |
|---|---|---|---|
| **GitHub Copilot (VS Code)** | VS Code | Proprietary (Copilot sub; BYOK works on Free) | **Chat/agent only.** Generic **Custom Endpoint** (`chat-completions`/`responses`/`messages`) — **Stable since 1.122 (May 28 2026)**; built-in **Ollama** provider (native `/api/*`); legacy `github.copilot.chat.customOAIModels` (OpenAI base URL). Inline completion NOT available locally. |
| **GitHub Copilot (Visual Studio / JetBrains)** | VS, JetBrains | Proprietary | Model picker; local/BYOK parity with VS Code is unverified from a primary source (treat as lagging). |
| **GitHub Copilot CLI** | Terminal | Proprietary | `COPILOT_PROVIDER_BASE_URL` [Ofox](https://ofox.ai/blog/github-copilot-byok-oai-compatible-api-setup/) + `COPILOT_MODEL` (+`COPILOT_PROVIDER_TYPE=azure/anthropic`); any OpenAI-compatible endpoint; requires tool calling + streaming; "for best results, use a model with a context window of at least 128k tokens." |
| **Continue.dev** | VS Code, JetBrains | Apache-2.0 | `provider: openai` + `apiBase`; [Continue](https://docs.continue.dev/customize/model-providers/top-level/openai) native `provider: llama.cpp`; `provider: ollama`; roles `chat/edit/apply/autocomplete/embed/rerank`. |
| **Cline** | VS Code | Apache-2.0 | "OpenAI Compatible" base URL + key + model ID; [Cline](https://docs.cline.bot/provider-config/openai-compatible) tolerant XML-ish tool parsing. |
| **Roo Code** | VS Code | Apache-2.0 | "OpenAI Compatible" base URL; **native tool calling only (≥3.37)**. |
| **Kilo Code** | VS Code | Apache-2.0 | OpenAI-compatible; XML tool-call option still present in advanced settings (later versions). |
| **Twinny** | VS Code, VSCodium | MIT | OpenAI-compatible chat; **`/infill` or FIM template** for completion; [Open VSX Registry](https://open-vsx.org/extension/rjmacarthy/twinny/3.17.6) llama.cpp/Ollama/LM Studio/Oobabooga presets. |
| **llama.vscode / llama.vim** | VS Code, Vim | MIT | **llama.cpp `/infill`** for FIM (required); `/v1/chat/completions` for chat/agent; `/v1/embeddings`. |
| **ProxyAI (formerly CodeGPT)** | JetBrains | Apache-2.0 | "Custom OpenAI" provider; FIM template for code completion; dedicated LLaMA C/C++ offline provider. |
| **Cursor** | Cursor (own) | Proprietary | "Override OpenAI Base URL" (+`/v1`); chat/agent; not local-friendly without a public/tunnel URL. |
| **Zed** | Zed (own) | GPL/Apache | `language_models.openai_compatible` with `api_url`, `available_models[].capabilities.{tools,images}`. |
| **Aider** | Terminal | Apache-2.0 | `OPENAI_API_BASE` + `OPENAI_API_KEY`, `--model openai/<name>`. [Aider](https://aider.chat/docs/llms/openai-compat.html) |
| **Void** | Void (own) | Apache-2.0 | OpenAI-compatible base URL (detailed behavior unverified). |
| **Tabby** | VS Code, JetBrains | Apache-2.0 (core) | `config.toml`: `kind="llama.cpp/completion"` (FIM via `prompt_template`), `kind="openai/chat"`, `kind="llama.cpp/before_b4356_embedding"`. [Tabby](https://tabby.tabbyml.com/docs/references/models-http-api/llamafile/) |
| **Tabnine, Qodo/Codium, Windsurf, Augment, Sourcegraph Cody, Pieces, Refact, Goose, OpenHands** | various | mixed | Most accept an OpenAI-compatible base URL for chat; FIM/autocomplete typically proprietary or model-specific (verify per-tool). Goose & OpenHands are agent frameworks consuming `/v1/chat/completions` + tools. |

### B. Exact wire contract per client (Copilot first)

**GitHub Copilot (VS Code).** Three configuration paths today:

- **Custom Endpoint provider (Stable since 1.122):** added via *Chat: Manage Language Models* → Add Models → Custom Endpoint. Supports per-model `apiType` ∈ `chat-completions` | `responses` | `messages`. The Insiders-era `chatLanguageModels.json` file used `vendor: "customendpoint"` with the same `apiType` selector.
- **Legacy `github.copilot.chat.customOAIModels` (still works in stable),** object keyed by model id:
```json
"github.copilot.chat.customOAIModels": {
  "my-model": {
    "name": "My Model",
    "url": "http://127.0.0.1:9931/v1/chat/completions",
    "toolCalling": true, "vision": false, "thinking": false,
    "maxInputTokens": 128000, "maxOutputTokens": 16000,
    "requiresAPIKey": false
  }
}
```
- **Built-in Ollama provider:** requires native endpoints `GET /api/version`, `GET /api/tags` (model list), `POST /api/show` (capabilities incl. context length, `tools`/`vision`). LM Studio issue #526 documents that emulating these is what unlocks Copilot's "Ollama" provider for non-Ollama servers. [GitHub](https://github.com/lmstudio-ai/lms/issues/526)

Copilot reads capability flags `toolCalling`, `vision`, `thinking`, `maxInputTokens`/`maxOutputTokens`. It sends standard OpenAI chat bodies with `messages`, `tools`, `tool_choice`, `stream:true`. A documented crash — microsoft/vscode issue #273482 ("OpenAI Compatible models return TypeError: Cannot read properties of undefined (reading 'cached_tokens')"), shows `TypeError: Cannot read properties of undefined (reading 'cached_tokens') at SX.push (…github.copilot-chat-0.33.2025102701…)` reproduced with LM Studio models in agent and ask mode — occurs when the streamed `usage` lacks `prompt_tokens_details.cached_tokens`. [GitHub](https://github.com/microsoft/vscode/issues/273482)

**Continue.dev** (`config.yaml`, schema v1): `provider: openai` + `apiBase: http://127.0.0.1:9931/v1` + `model` + `roles: [chat, edit, apply]`. For OpenAI-compatible non-chat completion: `useLegacyCompletionsEndpoint: true` forces `/v1/completions`. [Continue](https://docs.continue.dev/customize/model-providers/top-level/openai) Continue's native `llama.cpp` provider posts to `/completion` (singular), not `/completions` (issue #4991). `requestOptions.headers` carries auth; `capabilities: [tool_use, image_input]` can be declared.

**Cline / Roo Code:** Settings → "OpenAI Compatible" → Base URL (must include `/v1`), API key, model ID. Roo internally uses Anthropic message format then transforms to OpenAI `ChatCompletionTool`; it accumulates streamed fragments by `index`; finalizes on `finish_reason:"tool_calls"`. `parallelToolCalls:true` is the default.

### C. Inline autocomplete / FIM (highest priority)

**llama.cpp `/infill` contract (the target to implement):**
- `POST /infill`, fields: `input_prefix` (string, code before cursor), `input_suffix` (string, code after cursor), [GitLab](https://gitlab.informatik.uni-halle.de/ambcj/llama.cpp/-/blob/b2308/examples/server/README.md) `input_extra` (array of context chunks, prepended toward prompt start), `prompt` (optional raw text appended after the FIM middle marker), plus all `/completion` options; common params `n_predict`, `temperature`, `top_p`, `top_k`, `stop`, `samplers` (e.g. `["top_k","top_p","infill"]`), `cache_prompt:true`.
- Response: JSON with **`content`** (the completion — the only field clients require), plus `stop`, `tokens_predicted`, `timings`, etc. Streaming supported.
- The model's own FIM tokens are applied server-side from GGUF metadata, so clients send raw prefix/suffix. `--spm-infill` toggles SPM vs PSM ordering. [Debian Manpages](https://manpages.debian.org/unstable/llama.cpp-tools/llama-server.1.en.html)

**FIM control tokens by model family (verbatim — character precision matters):**

| Model | Tokens (verbatim) | Char notes | Order |
|---|---|---|---|
| **Qwen2.5-Coder** | `<\|fim_prefix\|>` `<\|fim_suffix\|>` `<\|fim_middle\|>` `<\|fim_pad\|>` `<\|repo_name\|>` `<\|file_sep\|>` | ASCII pipes (ids 151659–151664) | PSM: prefix·suffix·middle |
| **Code Llama** | `▁<PRE>` `▁<SUF>` `▁<MID>` `▁<EOT>` | `▁` = U+2581 (not ASCII underscore); ids 32007–32010 | PSM: `<PRE>`pre`<SUF>`suf`<MID>` (paper recommends PSM over SPM) |
| **DeepSeek-Coder (v1, 6.7b)** | `<｜fim▁begin｜>` `<｜fim▁hole｜>` `<｜fim▁end｜>` | `｜`=U+FF5C full-width pipe; `▁`=U+2581 | PSM: begin·pre·hole·suf·end |
| **DeepSeek-Coder-V2** | `<\|fim_begin\|>` `<\|fim_hole\|>` `<\|fim_end\|>` | ASCII pipe + ASCII underscore — **NOT byte-compatible with v1** | PSM |
| **StarCoder2** | `<fim_prefix>` `<fim_suffix>` `<fim_middle>` `<fim_pad>` `<file_sep>` `<repo_name>` | ASCII `<>` + underscore | PSM: prefix`<fim_suffix>`suffix`<fim_middle>` |
| **Codestral** | `[PREFIX]` `[SUFFIX]` (`[MIDDLE]`) | ASCII brackets; build via `mistral_common.encode_fim`, not by hand | SPM internal: `[SUFFIX]`suf`[PREFIX]`pre; API uses `prompt`+`suffix` |

Character-precision warnings: Code Llama and DeepSeek-Coder-v1 use the SentencePiece `▁` (U+2581) glyph, not an ASCII underscore; DeepSeek-Coder-v1 uses the full-width pipe `｜` (U+FF5C) while DeepSeek-Coder-V2 uses ASCII `|` + ASCII `_` (the two are not interchangeable — match the exact checkpoint); Codestral uses square-bracket `[PREFIX]`/`[SUFFIX]` (the widely-circulated `<PREFIX>`/`<SUFFIX>` angle-bracket claim is incorrect) and its FIM API is `POST /v1/fim/completions` with `prompt`+`suffix`.

**Per-client FIM behavior:**
- **llama.vscode / llama.vim:** `POST /infill`, reads `content`; defaults `cache_prompt:true`, `samplers:["top_k","top_p","infill"]`, `top_k:40`, `top_p:0.99`, `stream:false`; [DeepWiki](https://deepwiki.com/ggml-org/llama.vscode/3.2-llamaserver) warms cache with a fire-and-forget `n_predict:0` `/infill`. Requires llama.cpp (only server with `/infill`). Recommended launch: `llama-server -hf ggml-org/Qwen2.5-Coder-1.5B-Q8_0-GGUF --port 8012 -ub 1024 -b 1024 --ctx-size 0 --cache-reuse 256`.
- **Twinny:** OpenAI-compatible; per-model FIM template; CodeLlama uses `<PRE>{prefix}<SUF>{suffix}<MID>`, DeepSeek uses its FIM template; base (not instruct) models for FIM. "Twinny supports the OpenAI API specification so in theory any API should work."
- **Tabby:** `kind="llama.cpp/completion"`, `prompt_template="<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"` (Qwen2.5) [Tabby](https://tabby.tabbyml.com/docs/references/models-http-api/llamafile/) or `"<PRE> {prefix} <SUF>{suffix} <MID>"` (CodeLlama); endpoint **must NOT** include the `/v1` suffix.
- **Continue.dev autocomplete:** `roles:[autocomplete]`; `provider: llama.cpp` drives FIM; or `provider: openai` with a `template` Mustache string (`{{{prefix}}}`,`{{{suffix}}}`,`{{{filename}}}`,`{{{reponame}}}`,`{{{language}}}`). `autocompleteOptions`: `debounceDelay:250`, `maxPromptTokens:1024`, [Continue](https://docs.continue.dev/reference) `modelTimeout`, `maxSuffixPercentage:0.2`, `prefixPercentage:0.3`, `onlyMyCode:true`.
- **ProxyAI:** Custom OpenAI → Code Completions → "FIM Template (OpenAI)" + URL; [Medium](https://medium.com/@mitrut98/ghost-coding-on-prem-building-a-self-hosted-ai-copilot-for-intellij-or-any-jetbrains-ide-fdac377a10fd) uses `/v1/completions` or `/v1/chat/completions`.
- **Cline/Roo:** no ghost-text autocomplete; chat/agent only.

### D. Agentic tool-calling

OpenAI shape required: `tools:[{type:"function",function:{name,description,parameters}}]`, [GitHub](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md) `tool_choice` ∈ `auto|none|required|{type:"function",function:{name}}`. Streaming `delta.tool_calls[]` carry `index`, `id`, `function.name`, and incremental `function.arguments` string fragments; `finish_reason:"tool_calls"` terminates. [OpenAI API Reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events) **`function.arguments` MUST be a JSON-encoded string, not an object.** ggml-org/llama.cpp issue #20198 ("llama-server tool_calls returns arguments as JSON object instead of string, breaking OpenAI compatibility") documents that after the Autoparser refactoring (PR #18675), llama-server returned `arguments` as a parsed object (root cause in `common/chat.cpp` ~line 132: `{"arguments", json::parse(tool_call.arguments)}`), which crashes the official OpenAI Python SDK (Pydantic) with a `TypeError`. Your server must serialize arguments as a string.

- **Roo Code:** native only (≥3.37); transforms to OpenAI `ChatCompletionTool`; `parallelToolCalls:true` default; finalizes on `finish_reason:"tool_calls"` or stream end. Removal of XML tool calling broke some local stacks (issue #10319, SGLang gpt-oss 500 errors); rollback to 3.36.16 restores the XML selector.
- **Cline / Kilo Code:** historically XML-in-prompt tool calling parsed from text; tolerant of weak native support. The `native_tool_call_adapter` proxy exists specifically to translate Cline/Roo XML into OpenAI `tool_calls`.
- **Copilot agent:** native OpenAI tools via chat-completions; needs `toolCalling:true` on the model entry (a model that appears in chat but not agent mode usually has `toolCalling` missing/false).
- **llama.cpp tool support** requires `--jinja` (and often `--chat-template-file` for a tool-capable template; worst case `--chat-template chatml`). `chat_template_kwargs` (e.g. `{"enable_thinking":false}`), `parallel_tool_calls`, and `reasoning_format` (deepseek → `message.reasoning_content`) [Debian Manpages](https://manpages.debian.org/unstable/llama.cpp-tools/llama-server.1.en.html) are supported. [Fossies](https://fossies.org/linux/llama.cpp/tools/server/README.md) No client in scope strictly requires `/v1/responses`; Copilot's custom-endpoint can use `responses` or Anthropic `messages` but `chat-completions` suffices. Structured outputs (`response_format:{type:"json_schema"}`) are supported by llama.cpp via grammar but are not universally required by these clients.

### E. Model discovery & capabilities

- `GET /v1/models` clients read `id`, `object`, `owned_by`. Continue can use the special `AUTODETECT` model name. Roo/Cline mostly take an explicit model ID.
- Copilot's **Ollama** path reads context length and `tools`/`vision` from `POST /api/show` (microsoft/vscode issue #295659 shows Copilot's Manage Models UI expecting capability + context fields there, e.g. `262144` context, `Tools`/`Vision`). The OpenAI custom path takes `maxInputTokens`/`maxOutputTokens`/`toolCalling`/`vision` from settings, not from `/v1/models`.
- Zed reads `max_tokens`, `max_output_tokens`, `capabilities.{tools,images}` [Ofox](https://ofox.ai/blog/zed-editor-ai-configuration-guide-2026/) from its own settings, not the server.
- A single advertised model is fine for most clients; multi-model is optional. Non-standard capability fields on `/v1/models` are largely ignored — capabilities are configured client-side.

### F. Cross-cutting compatibility pitfalls

- **Trailing usage chunk:** when `stream_options:{include_usage:true}`, emit a final chunk with `choices:[]` and a populated `usage`, [LiteLLM](https://docs.litellm.ai/docs/completion/usage) then `data: [DONE]`. All non-final chunks should carry `usage:null` (per OpenAI's documented streaming shape and LiteLLM docs).
- **Copilot `cached_tokens`:** include `usage.prompt_tokens_details.cached_tokens` or Copilot's custom-OAI path throws `Cannot read properties of undefined (reading 'cached_tokens')` (vscode #273482).
- **finish_reason after tool calls:** must be `"tool_calls"` on the terminating chunk in streaming, else agent loops terminate early [GitHub](https://github.com/open-webui/open-webui/issues/21768) (the open-webui #21768 pattern: "finish_reason incorrectly returned as 'stop' after streaming tool_calls").
- **First delta with role:** emit an initial `delta:{role:"assistant",content:""}` chunk (matches OpenAI's documented first streamed event).
- **`data: [DONE]` terminator** is expected by OpenAI-style consumers; always send it last. LiteLLM #25389 shows consumers that stop at `finish_reason` lose the trailing usage chunk — keep the stream open until `[DONE]`.
- **CORS / preflight:** browser/webview clients send `OPTIONS` preflights and an `Authorization` header; respond to `OPTIONS` with `Access-Control-Allow-Origin`, `Access-Control-Allow-Methods: GET,POST,OPTIONS`, `Access-Control-Allow-Headers: Content-Type, Authorization`. Ollama's default of restricting origins/headers is a documented friction source ("Request header field 'authorization' is not allowed by Access-Control-Allow-Headers in preflight response").
- **Path / `/v1` differences:** some clients append `/v1` (Continue `openai`, Cline, Zed), some must NOT (Tabby `llama.cpp/completion`, llama.vscode `endpoint_chat` excludes `v1`). Continue's `llama.cpp` provider uses `/completion` singular. Support both trailing-slash and non-slash forms.
- **Keep-alive / timeouts:** long prefill needs SSE heartbeats (you already emit these) and generous read timeouts (llama.cpp server default is 600s; Continue defaults `requestOptions.timeout` to tens of seconds — local guides raise it to 60000 ms for CPU).
- **gzip:** accept but don't require; some clients send `Accept-Encoding: gzip`.
- **arguments-as-string** (Section D) is the single most damaging non-spec deviation.

### G. Adjacent features

- **Embeddings:** `POST /v1/embeddings` (`input` string or array, `model`, `encoding_format`). Used by Continue (`roles:[embed]`), Twinny (workspace embeddings, default `all-minilm:latest`), llama.vscode (semantic re-rank). Response must be OpenAI `data:[{embedding,...}]` shaped; llama.cpp's native `/embedding` is non-OAI [GitHub](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) , so clients want `/v1/embeddings`.
- **Reranking:** llama.cpp exposes `POST /v1/rerank` [GitHub](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) (also `/rerank`, `/reranking`) requiring `--reranking`/`--pooling rank`; request `{model,query,documents,top_n}`, response `{results:[{index,relevance_score}]}`. **Continue expects a `data` array and errors on llama.cpp's `results` shape** (continue #6478: "Expected 'data' array but got: ['id','model','usage','results']") — a real interop gap to document or shim.
- **Prompt / KV cache:** `cache_prompt:true` and `--cache-reuse N` reuse common prefixes — essential for repeated-prefix latency (chat turns, FIM). `--cache-reuse` has regressed before (llama.cpp #15082) — pin a known-good build.
- **Vision:** chat `content` parts with `image_url`/base64; llama.cpp multimodal needs `--mmproj`; [GitHub](https://github.com/DelftSolutions/vscode-llama-copilot) clients gate on `vision:true` / `capabilities.images`. Used by Cline/Roo screenshots and ProxyAI image chat. Per the llama.cpp server README, "A client must not specify [media] unless the server has the multimodal capability. Clients should check /models or /v1/models for the multimodal capability before a multimodal request."

## Recommendations

**Must-have for broad compatibility (do these first):**
1. **Implement `POST /infill`** with `input_prefix`/`input_suffix`/`input_extra`/`prompt`/`n_predict`/`cache_prompt`, returning `content`. Unlocks llama.vscode, llama.vim, Twinny (llama.cpp mode), Tabby, Continue `llama.cpp` autocomplete. *This is your biggest gap.*
2. **Serialize `tool_calls.function.arguments` as a JSON string** (never an object). Unlocks Roo Code, Copilot agent, any OpenAI-SDK consumer. Acceptance benchmark: the OpenAI Python SDK must parse your tool response without `TypeError`.
3. **Streaming tool-call correctness:** `delta.tool_calls` with `index`/`id`/`function.name`/incremental `arguments`, terminating `finish_reason:"tool_calls"`. Unlocks all agent modes.
4. **`stream_options.include_usage` trailing chunk** with empty `choices` + populated `usage`, always ending with `data: [DONE]`, and include `usage.prompt_tokens_details.cached_tokens`. Unlocks Copilot custom-endpoint without crashes.
5. **CORS/OPTIONS handling** allowing `Authorization` + `Content-Type`. Unlocks webview/browser clients.
6. **Tolerant routing:** accept both `/v1/...` and bare paths, with and without trailing slash; accept `/completion` and `/completions`.

**High-value:**
7. **Emulate the Ollama native API** (`GET /api/version`, `GET /api/tags`, `POST /api/show` with `capabilities` incl. tools/vision + context length, and `POST /api/chat`/`/api/generate`). *Downgraded from earlier priority:* because the OpenAI Custom Endpoint provider reached VS Code Stable in 1.122 (May 28, 2026), a clean OpenAI-compatible surface now covers Copilot chat/agent. Ollama emulation is still worthwhile to support older VS Code versions and tools hard-coded to Ollama's native endpoints, but it is no longer on the critical path for current Copilot.
8. **`POST /v1/rerank`** (`{query,documents,top_n}` → `{results:[{index,relevance_score}]}`) and consider also returning a `data` array alias for Continue. Unlocks RAG / "chat with codebase."
9. **`cache_prompt` + prefix reuse** and SSE heartbeats during prefill (you have heartbeats; add prefix reuse).
10. **Advertise capabilities** in `/v1/models` and (if Ollama-emulating) `/api/show` so agent modes light up.

**Nice-to-have:**
11. **Vision** via `image_url` content parts (needs mmproj).
12. **Anthropic `/v1/messages`** and **OpenAI `/v1/responses`** shims for Copilot's other apiTypes and for Claude-shaped clients (Claude Code, etc.).
13. **Per-model FIM template registry** (Qwen / CodeLlama / DeepSeek v1 & V2 / StarCoder2 / Codestral) if you also expose `/v1/completions`-with-`suffix` for clients that don't use `/infill`.
14. **`/props` / `/v1/models` context-length reporting** so clients auto-size prompts.

**Staged rollout:** Ship (1)–(6) → validate against Continue (autocomplete + chat + agent), Twinny (FIM), Roo Code (native tools), Copilot Custom Endpoint (chat/agent). Then (7)–(10) → validate Copilot Ollama provider + RAG. Then (11)–(14).

## Exact config snippets

**GitHub Copilot (VS Code ≥1.122) — Custom Endpoint (preferred):** *Chat: Manage Language Models* → Add Models → **Custom Endpoint** → display name, API key (any string if none), Base URL `http://127.0.0.1:9931/v1`, model ID, apiType `chat-completions`. For older stable builds, use the legacy object in `settings.json`:
```json
"github.copilot.chat.customOAIModels": {
  "local-qwen": {
    "name": "Local Qwen (llama.cpp)",
    "url": "http://127.0.0.1:9931/v1/chat/completions",
    "toolCalling": true,
    "vision": false,
    "thinking": false,
    "maxInputTokens": 32768,
    "maxOutputTokens": 8192,
    "requiresAPIKey": false
  }
}
```
(For Copilot's built-in **Ollama** provider, run an Ollama-emulation layer on `:11434` and use Chat: Manage Language Models → Add → Ollama. Inline completions cannot be served locally in any case.)

**Continue.dev — `~/.continue/config.yaml`:**
```yaml
name: Local llama.cpp
version: 1.0.0
schema: v1
models:
  - name: local-fim
    provider: llama.cpp
    model: your-model.gguf
    apiBase: http://127.0.0.1:9931
    roles: [autocomplete]
    autocompleteOptions:
      debounceDelay: 250
      maxPromptTokens: 1024
  - name: local-chat
    provider: openai
    model: your-model
    apiBase: http://127.0.0.1:9931/v1
    apiKey: sk-local
    roles: [chat, edit, apply]
```

**Cline / Roo Code (VS Code settings UI):**
- API Provider: **OpenAI Compatible**
- Base URL: `http://127.0.0.1:9931/v1`
- API Key: `sk-local` (any string if no auth)
- Model ID: `your-model`
- (Roo Code ≥3.37: model must support native tool calling, or roll back to 3.36.16 for the XML option.)

**Twinny (Providers → Code/FIM provider):**
- Provider: `llamacpp`
- Hostname/port: `127.0.0.1` / `9931`
- FIM endpoint path: `/infill`
- FIM Template: match the model (e.g. Qwen2.5-Coder / DeepSeek / CodeLlama)

**Tabby — `~/.tabby/config.toml`:**
```toml
[model.completion.http]
kind = "llama.cpp/completion"
api_endpoint = "http://127.0.0.1:9931"   # no /v1
prompt_template = "<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"

[model.chat.http]
kind = "openai/chat"
api_endpoint = "http://127.0.0.1:9931/v1"
```

## Caveats / open risks (2026 moving targets)
- **Copilot Custom Endpoint (OpenAI) reached Stable in VS Code 1.122 (May 28, 2026).** This is recent; behavior on older VS Code (and the exact apiType handling) may differ. The legacy `github.copilot.chat.customOAIModels` object is slated to change to an array form (microsoft/vscode issue #277102) — track this.
- **Copilot inline completion remains closed to local models** — a stable, documented limitation as of 2026 ("Inline suggestions and next edit suggestions still require a GitHub sign-in"). The only local autocomplete in VS Code is via third-party `InlineCompletionItemProvider` extensions (Continue, Twinny, llama.vscode).
- **Roo Code removed the XML tool-calling selector in 3.37** and forces native; this broke some local stacks (issue #10319). A fallback may return in future versions — verify per release.
- **llama.cpp build-dependent behavior:** the `tool_calls` arguments-as-object regression (#20198, from PR #18675) and `--cache-reuse` regressions (#15082) mean you should pin a known-good commit and add regression tests for both.
- **llama.cpp rerank path aliases** (`/rerank` vs `/v1/rerank` vs `/reranking`) have shifted across releases; Continue expects a `data` array, not `results` [GitHub](https://github.com/continuedev/continue/issues/6478) (#6478). Reranker score quality also varies with GGUF conversion/quantization (#16407).
- **Visual Studio / JetBrains Copilot** local-model parity with VS Code is unverified from a primary source — treat as "lagging/uncertain" and test directly before claiming support.
- **Copilot CLI BYOK** (GitHub Changelog, Apr 7 2026) requires tool calling + streaming and recommends ≥128k context; `COPILOT_OFFLINE=true` enables air-gapped use with a local provider.

*Primary sources cited inline include: VS Code docs (code.visualstudio.com/docs/agent-customization/language-models) and the v1.122 release notes; GitHub Changelog (Apr 7 & Apr 22, 2026) and GitHub Docs BYOK pages; ggml-org/llama.cpp `tools/server/README.md`, `docs/function-calling.md`, and issues #20198, #15082, #16407, #16498, #21415; ggml-org/llama.vscode repo/wiki and DeepWiki; Continue.dev docs (autocomplete, yaml-reference, openai provider) and issues #4991, #2330, #6478; Roo Code docs and issues #4047, #10319; Cline docs; Tabby docs (llama.cpp/llamafile/model config); Twinny repo/docs; ProxyAI repo/docs; microsoft/vscode issues #273482, #277102, #295659; Hugging Face model cards/papers for Qwen2.5-Coder, Code Llama, DeepSeek-Coder, StarCoder2, and Codestral. Claims about Visual Studio/JetBrains Copilot parity and the Void editor are flagged as unverified.*