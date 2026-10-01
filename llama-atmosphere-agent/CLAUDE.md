# CLAUDE.md — `llama-atmosphere-agent/` (local coding agent with Atmosphere)

Guidance for working in this directory. Claude Code loads it when files here are read; the
repository-wide rules are in [`../CLAUDE.md`](../CLAUDE.md), which keeps a short summary of what
this project means for the rest of the repository (release asset, CI gates, version bump).


A **copy-and-run general-purpose terminal agent** (Claude Code / OpenCode reduced to the essentials, offline)
that pairs [Atmosphere](https://github.com/Atmosphere/atmosphere)'s built-in OpenAI-compatible
agent runtime with this project's `OpenAiCompatServer`. Like `android-llmservice/` it is a
**standalone Maven project, NOT a reactor module** — it is an application, it needs Java 21
(Atmosphere's floor) while the core stays Java 8, and it is built without a parent so the folder can be
copied out and run on its own. **Its version is the core's** (`5.2.0-SNAPSHOT` on `main`), and
`check-natives.py` fails when the two differ: `versions:set` does not reach this pom, so a release bump
that forgot it would otherwise publish an agent naming the wrong core. `llama.version` defaults to
`${project.version}`; CI still passes `-Dllama.version=<reactor version>` (the same value).

**It is published to Maven Central** as `net.ladenthin:llama-atmosphere-agent`: the thin jar (with
`Main-Class`, so `jbang <coordinates>` starts it), its sources and javadoc jars and the pom, signed, by
a step of its own in `publish-snapshot` / `publish-release` right after the reactor deploy
(`-P release deploy`; the `release` profile here mirrors the parent's). It resolves the core and the
natives jars from the local repository, where that reactor deploy has just installed them.

**The natives are a plain `runtime` dependency on `llama-platform`**, not a profile: they are part of
the published pom, and a consumer (JBang, Maven, Gradle) must get them however it treats profiles —
without them the agent fails at the first load. CI installs only the core classes and the
`llama-platform` pom (`build-core` with `modules: llama,llama-platform`), because no natives jar exists
before the `package` job, and passes `-Dllama.natives=none`: that activates the `no-natives` profile,
whose `dependencyManagement` excludes everything `llama-platform` names, so only its pom is resolved.
The model-backed job loads a downloaded library through `-Dnet.ladenthin.llama.lib.path`. The
`gpu-natives` profile (`-Dllama.classifier=<natives jar>`) adds a GPU backend next to the CPU natives.

**Release asset: the agent jar WITHOUT the core.** `mvn -P assembly package` (the pom's `assembly`
profile, descriptor `src/assembly/agent-jar.xml`) builds
`llama-atmosphere-agent-<llama.version>-jar-with-dependencies.jar` — named after the **core** version
it was built against (the agent's own version is the same), because it only runs next to that core.
It excludes `net.ladenthin:llama` **with its whole runtime graph** (`useTransitiveFiltering`: Jackson 2,
slf4j-api, and Jackson 3's `jackson-annotations`, which resolves through the core's trail) plus
`jspecify` and `slf4j-simple`, all of which every core fat jar already bundles — so the asset is ~14 MB
(most of it Jetty, the ACP SDK with Reactor, and the Atmosphere console pages) instead of hundreds, the natives are not in the release twice, and there are never two SLF4J providers.
The manifest's `Class-Path` names the four `llama-<v>-all-<os>-<arch>-…` fat jars and then the default
`llama-<v>-jar-with-dependencies.jar`, so `java -jar` works when the agent lies next to any of them
(missing entries are ignored; note that a manifest `Class-Path` is honoured under `java -cp` too).
**Rename a core fat jar and this list must follow** — `smoke-agent-linux` is what notices.
CI wiring (`publish.yml`): the model-free job builds it, writes the `.sha256` and uploads artifact
`llama-atmosphere-agent-jar`; `github-snapshot` / `github-release-signed` download it into the asset
directory next to `llama-fatjars`, so `sign-fatjars.sh` signs it (`*-jar-with-dependencies*.jar`) and
the one upload attaches it. **`smoke-agent-linux`** (`.github/smoke-agent-jar.sh`) runs the asset the
way the README tells a user to — `java -jar` next to the real `all-linux-x86-64` fat jar — and checks:
bytecode ≤ 65 (Java 21, unlike the core's 52 — with one `--allow` for JLine's FFM terminal provider `org/jline/terminal/impl/ffm/*`, 25 classes shipped as Java 22 bytecode that JLine discovers through `META-INF/jline/providers/ffm` and never loads on 21, where it picks its JNI provider; the first CI run of the smoke caught them), that the jar started **alone** fails with
`NoClassDefFoundError: net/ladenthin/llama/LlamaModel` (i.e. it really carries no core), `--help`, a
one-shot `2 + 2` answer and a `read_file` round that must surface a marker from `--workspace`, all on
the cached `TOOL_MODEL_NAME` with `--ngl 0`; then `--web` on port 0 (the console is `401` without the
token, the token link answers `302` with the `jllama_agent` cookie, the console page is served with it)
and `--acp` through `.github/smoke/agent_acp_smoke.py` (standard-library Python playing the editor:
handshake, modes, a streamed `4`, the announced commands, a `read_file` tool card that brings a marker
back, a clean exit when stdin closes, and nothing but JSON-RPC on stdout). **All three agent jobs gate
both publish jobs** (model-free, model-backed integration, smoke).

**The console pages come from a jar the assembly otherwise leaves out.** Atmosphere's prebuilt AI
console lives in `atmosphere-spring-boot-starter` under `META-INF/resources/atmosphere/console/`. The
pom depends on that starter with a `*:*` exclusion (no Spring on the classpath), the assembly's main
dependency set excludes it, and a second dependency set unpacks **only** that resource path. Remove
either half and the jar either grows by the whole starter or `--web` starts with nothing to open
(`WebConsole.available()` says so at startup).

**What Atmosphere is, for this purpose.** `org.atmosphere:atmosphere-ai` (4.0.71) ships
`BuiltInAgentRuntime` + `OpenAiCompatibleClient`: a zero-framework OpenAI client that *always*
streams (`stream:true`), accumulates `delta.tool_calls` by `index`, executes `ToolDefinition`
executors, re-submits the conversation (assistant `tool_calls` message **without** a `content` key,
then one `role:"tool"` message per call with `tool_call_id` + `name`), and loops until
`finish_reason` is not `tool_calls`. It reads `LLM_BASE_URL`/`LLM_MODEL`/`LLM_API_KEY` or takes
`AiConfig.configure(mode, model, apiKey, baseUrl)`; `GET /models` is best-effort; the Responses API
is used only when the base URL contains `api.openai.com`; `tool_choice`/`parallel_tool_calls`/
`response_format` are not sent. It runs headless — `runtime.execute(AgentExecutionContext,
StreamingSession)` — so no Spring Boot, servlet container or `@Agent` scanning is involved; its
built-in `FileSystemTools` resolve the `AgentFileSystem` from `StreamingSession.injectables()`,
which is how the tools are confined to a workspace. The `@Agent`/`@AiTool` annotations and the
Spring Boot starter are a deployment layer on top of the same runtime.

**Verified compatibility (verdict A — works unchanged).** Two test layers, both in the project:

- `AtmosphereWireContractTest` + `LocalAgentTest` — **model-free, every PR, seconds**: the *real*
  `OpenAiCompatServer` (routing, bearer auth, `/v1/models`, SSE framing) over a loopback socket with
  a `ScriptedBackend` replaying llama.cpp-shaped chunks (role delta, `tool_calls` deltas with
  `index`/`id`/`name` and fragmented `arguments`, `finish_reason:"tool_calls"`). Pins: one tool
  round; four rounds incl. a parallel pair with interleaved fragments and the whole history kept;
  chunk-by-chunk streaming and history replay; `temperature`/`max_tokens` on the wire; 401 on a
  wrong key before the backend is reached; and the **one known gap** — an engine failure *after* the
  stream started is an SSE `data: {"error":…}` under HTTP 200 (upstream llama-server does the same),
  which Atmosphere's parser ignores (it reads only `choices[0]`), so the turn completes with the
  text so far instead of erroring. That is a SHOULD for Atmosphere's `OpenAiCompatibleClient`, not
  for this project.
- `AtmosphereToolLoopIntegrationTest` — **model-backed, CI only** (`test-java-llama-atmosphere-agent-integration`,
  a publish gate): the same loop against the cached Qwen2.5-1.5B tool model
  through the downloaded Linux natives — plain chat, streaming (≥ 2 chunks), a tool call whose result
  is answered, a read→write→read loop that changes a temp file. Self-skips without the GGUF.

**The one core change this needed:** `OpenAiBackend`, `ChunkSink` and
`OpenAiCompatServer(OpenAiBackend, OpenAiServerConfig)` are now **public** (they were the
package-private test seam). A sibling module cannot otherwise drive the real server without a model;
the alternative — a same-named package in the sibling's test tree — is a split package that breaks
the moment anything runs on the module path.

**Layout.** `AgentOptions` (CLI parsing, pure), `AgentRunner` (the whole Atmosphere wiring, ~40
lines: `AiConfig.configure` → `BuiltInAgentRuntime` → `AgentExecutionContext` + `ToolLoopPolicies`),
`ConsoleSession` (streams to stdout, prints `⚙ tool {args}` / `↳ result`, supplies the
`WorkspaceAgentFileSystem` via `injectables()`), `ShellTool` (opt-in `run_command`, `sh -c` /
`cmd /c` starting in the workspace, timeout kills the process tree, output tail-truncated), `LocalAgent`
(`--base-url` = external server, `--model` = in-process `LlamaModel` + loopback `OpenAiCompatServer`
with `enableJinja()` and `setLogVerbosity(2)` by default — llama.cpp logs to **stderr**, the console the
streamed answer shares, so the per-request `slot …` INFO lines would interleave with it; `--log-verbosity <n>`
/ `--verbose` override — one-shot `--prompt` or a `you>` REPL with `/clear` `/exit`). `.mvn/jvm.config`
pins `-Dstdout.encoding=UTF-8 -Dstderr.encoding=UTF-8` for the `mvn exec:java` JVM: on Windows,
`common_init()` switches the console to UTF-8 (`SetConsoleOutputCP(CP_UTF8)`) after the JVM fixed its
stdout encoding from the old code page, which turned umlauts/emoji in answers into `�`/`?`. It needs a
sibling **`.mvn/jvm.config.license`** (the same two SPDX lines as the one next to the root `.mvn/jvm.config`):
a `jvm.config` takes no comments, so REUSE can only read its metadata from that file, and without it the
`REUSE Compliance Check` job fails on `main` — which is how it was found, the PR run having been cancelled.
Spotless (palantir) is configured in its own pom; the model-free CI job runs `spotless:check`.

**The REPL layer (commands, approval, status line, rendering).** Eight small classes and one dependency
(`org.jline:jline`, one jar, no transitive deps):
`SlashCommands` (a line starting with `/` whose first word names a command is handled locally —
`/help /status /tools /mode /compact /clear /exit`; **an unknown `/command` goes to the model**, which
is why no escape syntax is needed for `/usr/bin/…`), `ApprovalMode` + `ConsoleApprovalStrategy`
(`[y]es/[n]o/[a]uto` per gated call), `StatusLine`, and `Ansi` + `MarkdownConsole`. Four points that
are decisions, not details:

1. **The approval gate is Atmosphere's, not ours.** `AgentRunner.approval(strategy, policy)` attaches
   `ToolApprovalPolicy.custom(...)` (gating `run_command`, `write_file`, `edit_file`, `delete`,
   `rename` — reading tools never ask) and a `ConsoleApprovalStrategy`; `ToolExecutionHelper` then
   blocks the tool loop before the executor runs and turns a denial into the tool result
   `{"status":"cancelled","message":"Action cancelled by user"}` for the model. Do not reimplement
   that message. `ApprovalWireTest` pins both halves over the real server. **The reading tools
   (`ls`, `read_file`, `glob`, `grep`) never ask, and that is a decision, not an omission**: gating
   them would make the question so frequent it stops being read. Because the gate is a list of
   *names*, a tool upstream adds or renames would drop out of it and then run unasked — so
   `READ_ONLY_TOOLS` names the other half explicitly and
   `ConsoleApprovalStrategyTest.everyOfferedToolIsEitherGatedOrDeclaredReadOnly` asserts every offered
   tool is in exactly one of the two sets, and that neither set names a tool nobody offers. Same class
   as the stale `spotbugs-exclude.xml` entries: an allowlist that silently stops matching.
2. **One-shot (`--prompt`) denies a gated call** instead of auto-approving it — `--auto` is the
   deliberate opt-in. Atmosphere itself fails closed when no strategy is wired, and this keeps that
   direction: an unattended run must not be the most permissive one.
3. **`AgentTerminal` has exactly two implementations, chosen once at startup, and `--plain` picks the
   line-oriented one on purpose.** `LocalAgent.usesFullTerminal(options, interactive)` is the single
   place that decides: the cursor-controlling console needs someone typing **and** permission to move
   the cursor, and `--plain` withholds the second even on a real terminal. That is not a fallback but
   a supported mode — for a session that is piped, logged, recorded, or carried by something that
   forwards lines rather than a screen. It gives up the pinned block, the spinner, history/completion
   and typing-during-a-turn (`PlainTerminal.hasPendingInput()` is always false), and gains being
   correct when the output is a file. A normal SSH session needs none of this: a remote terminal
   reports its size and handles cursor control like a local one. `JLineTerminal` (a real
   terminal: line editing, history, Tab completion of the command names, a status line pinned to the
   bottom via JLine's `Status`, single-key answers through `enterRawMode`, streamed output via
   `LineReader.printAbove` so the bottom block stays put) and `PlainTerminal` (a `PrintStream` plus a
   `BufferedReader`: no cursor control at all, correct when the output is a file). `JLineTerminal.open`
   returns **null** instead of throwing when there is no usable terminal — piped input, a dumb
   terminal, a missing native provider — and the caller falls back. Every test drives `PlainTerminal`,
   which is why none of them needs a TTY. Verified on Windows: JLine picks the `windows-vtp` provider,
   so ANSI works there without the registry caveat.
4. **The answer is rendered append-only, one completed line at a time** (`MarkdownConsole`). Redrawing
   on every token is what produces the known overdraw/truncation bugs in the Ink/Bubble-Tea based
   clients and breaks when the output is piped. Only headings, bullets, fences and inline
   `**bold**`/`` `code` `` are handled; italics deliberately are not (`*` is more often a glob than
   emphasis). Colour is decided once in `Ansi.detect()` — `CLICOLOR_FORCE`, then `NO_COLOR`, then
   `TERM=dumb`/`CLICOLOR=0`, else "is a terminal" via `Console.isTerminal()` (reflective: JDK 22+;
   below that `System.console() != null`).
5. **`/loop` keeps its state in a file, not in the context, and stops on a text marker** (`TaskLoop`,
   `LoopOptions`). Every step re-sends the task verbatim and drops the history, so the context cannot
   grow (Claude Code's ralph-wiggum plugin does the same, and `AGENT-LOOP.md` in the workspace is the
   memory). The stop signal is a line that is **exactly** `<<TASK_COMPLETE>>`, never a substring —
   deliberately **not** a "done" tool: below 7B a model emits a malformed tool call far more often
   than a malformed line, and mini-SWE-agent's SWE-bench results come from exactly this plain-sentinel
   design. `--check '<cmd>'` re-verifies the claim and feeds a failure back. Four guards, none of them
   trusted to the model: step cap, wall-clock budget, stall detection (three steps with no file change
   and no tool call), interval. **Order matters and a test pins it**: the marker is checked *before*
   the stall detector, because the step that only answers "done" changes nothing and would otherwise
   be reported as no progress.
6. **Three of the file tools are this project's, not Atmosphere's** (`WorkspaceTools`, `TextEdits`,
   `WorkspaceSearch`) — `read_file`, `edit_file`, `grep`. They are **replacements, not additions**:
   two tools that both claim to read a file is the worst case for tool selection. They call the same
   `AgentFileSystem`, so workspace confinement, path validation and size limits stay Atmosphere's.
   Each replacement has a measured reason, and all three are pinned by tests:
   - `edit_file`: the framework matches against the **raw** file content, so a model's LF text never
     matches a CRLF file — on Windows *every* edit fails silently. `TextEdits` normalizes before
     matching and restores the file's own ending and byte-order mark. It also shows the nearest lines
     on a miss and the line numbers on an ambiguous match (a failed edit drops the eventual success
     rate from 90.5 % to 57.2 %), offers `replace_all`, and applies a batch of edits **all-or-nothing**
     — a deviation from every shipping agent, which apply sequentially and leave a half-edited file.
   - `grep`: the framework walks **alphabetically** with one global 2-second deadline and one global
     500-hit budget, so `.git`, `target` and `node_modules` consume both before `src` is reached.
     `WorkspaceSearch` excludes them, groups by file with line numbers, and **states** truncation.
   - `read_file`: `offset`/`limit` and numbered lines (whole-file reads measure 12.7 % against 18.0 %
     task success in the SWE-agent ablations). The numbers are display only, which both the tool
     description and the system prompt say — leaked line numbers in `old_string` are a known failure.
   - `edit_file` **refuses a file that was not read** in this session (`WorkspaceTools.ReadTracker`).
     Not a staleness check: an exact unambiguous match is safe regardless; this catches the model
     inventing the text.
   **Rejected on evidence, do not add later without new numbers:** a unified-diff/patch tool (Meta's
   ablation: search-replace 42–53 % vs 26–30 % unified diff vs 20–26 % line diff on one model; a 7B
   model collapses 54 → 33 → 14 %), fuzzy matching (turns a loud miss into a silent wrong-place edit),
   an embedding index (Cursor's production effect is +0.3 %), and LSP tools (the one isolation study
   finds them token-negative and *worse* at multi-file rename, because renames touch comments and
   strings that semantic references exclude).
7. **The session transcript (`Transcript`) is not the conversation the model is sent, and must not be
   merged with it.** The model's history is rewritten by `/compact` — a summary replaces the turns —
   and has never carried a timestamp; the transcript only grows and stamps every entry. `/compact`
   adds a note to it and changes nothing else, `/clear` empties it (the command means "forget this
   session"), `/save [name]` writes it into the workspace, and `--transcript <file>` appends live so a
   killed session still leaves what it had — that write failing is swallowed, because a record that
   exists to survive a bad ending may not cause one. **A list, not a map keyed by the timestamp**: a
   tool result and the answer after it regularly share a millisecond and a map would drop one
   silently; insertion order already is time order. `ToolCallLog` stays as the separate, hard-cut
   receipt for `/calls` — it answers "did that really run", which prose cannot.
   **`/load` replays only `USER` and `AGENT` entries** as messages: a tool result outside its round is
   not something a chat template has a place for, and inventing a shape for it would be worse than
   letting the model call the tool again. The parser treats a line without a stamp as a continuation
   of the entry above it, because an entry is not a line — an answer keeps its newlines when written,
   and reading line by line would turn one answer into several. A file that is not a transcript yields
   **no** entries rather than one wrong one, since anything it yielded would be replayed to the model
   as if it had been said.

   **`JLineTerminal.open` refuses when there is no console, and that was a red build for months.** A system
   terminal takes over the process's standard input, and where there is no console that input belongs to somebody
   else. Inside a **Surefire fork it is the channel Surefire talks over**: a test that drives the agent
   interactively reached `open()`, JLine grabbed the channel, and the run ended with
   `[SUREFIRE] std/in stream corrupted` — **every test green, the build red**, which is exactly the shape that
   hides. It also made `LocalAgentTest` take 23.6 s instead of 2.0 s, because it was blocking on reads that were
   never going to arrive. The check is the same two-step `Ansi` uses for colour: `Console.isTerminal()` where it
   exists (JDK 22+, where `System.console()` returns a console even for redirected streams), otherwise the mere
   presence of a console. Piped input lands here too and has always been served by the plain console, so nothing
   else changes.

8. **Tool calls are carried into the conversation as a text note, and logged for `/calls`.**
   `LocalAgent.withToolNotes` prefixes each turn's answer in the history with
   `(tools I actually ran this turn: <tool> <args> -> <result, cut at 400 chars>)`, and `ToolCallLog`
   keeps the same data for the `/calls` command. **Why it is a note and not real `tool_calls`
   messages:** `AbstractAgentRuntime.assembleMessages` rebuilds every history entry as
   `new ChatMessage(h.role(), h.content())` — the tool-call array and the tool-call id never leave the
   framework, so protocol-faithful replay through `context.history()` is impossible; content is what
   survives. **Why it exists at all:** with only user text and assistant prose in the history, a 4B
   model stopped calling tools after the third turn of a real session and *described* the work instead
   — inventing JUnit tests, a Maven build and a `.bat` script, complete with exit codes, while the
   workspace stayed empty. The system prompt also forbids claiming an action without the call.
   **Placement was found by failing twice, so do not "simplify" it:** in front of the assistant's
   answer made the model copy the record into its own replies (the user saw `(tools I actually ran
   this turn: …)` as the first line of an answer); real `tool_calls` messages are impossible (see
   above); a mid-history system message is cleanest but Mistral's template requires strict
   user/assistant alternation and Gemma has no system role. It therefore rides in front of the **next
   user message**, which every template accepts.
   Pinned by `LocalAgentTest.aToolCallStaysInTheHistorySoTheNextTurnSeesItHappened`.
   **Live feedback while a turn runs** (`LocalAgent.activityLine`, `ShellTool`'s line-by-line output):
   the block's first row names the running tool and its own elapsed time, and shell output is printed
   as it arrives. **The turn runs on its own thread, and it has to:** Atmosphere's `execute()` is
   synchronous — it returns only once the whole turn including every tool round is done — so running
   it on the console thread leaves nobody to refresh the line, and the block sits on
   "… waiting for input …" for the entire turn (exactly the symptom that was reported). The approval
   prompt then reads a key in raw mode on that worker thread while the console thread redraws four
   times a second, so `TurnActivity` pauses the redraw for as long as the question is open. Reading the pipe incrementally is not only cosmetic — an unread pipe blocks the child once
   it is full, which on Windows is roughly 4 KB.
9. **The context number in the status line is an estimate, marked `~`, and it moves during the turn.**
   llama.cpp emits its usage chunk only when the client sets `stream_options.include_usage`, and
   Atmosphere's client does not; `ConsoleSession.usage()` takes the real count when one arrives,
   otherwise `LocalAgent.estimateTokens` uses four characters per token. The window size is
   `--ctx-size` (in-process) or the server's `/props` (`ServerProps`), and is omitted rather than
   guessed when neither answers. **The state row is a `Function<ConsoleSession, String>`, not a
   string**, and `awaitWithActivity` asks it again on every redraw: it used to be rendered once before
   the turn and handed over fixed, so the figure stood still through every tool round and only moved
   at the next `you>` — which is exactly when it no longer helps anyone decide whether to `/compact`.
   `LocalAgent.liveTokens` adds `ConsoleSession.producedChars()` (streamed text **plus** every tool
   call and result — all of it is in the prompt of the next model call of the *same* turn) to what the
   request carried when it was sent, and yields to the server's own count as soon as one arrives.
   `TaskLoop` passes a constant function, and its step label must be copied into a local first: a
   lambda may not close over the loop counter.
10. **One call to `AgentTerminal.line` is one screen line**, and `ConsoleSessionTest` is what defends
   it. The pinned block is reserved in **lines**, so a single "line" carrying twenty newlines moves the
   screen twenty rows further than the terminal accounted for and the block is then drawn across the
   output — reported twice, both times from a `write_file` call whose `content` argument was the file.
   `ConsoleSession.describeArguments` folds and cuts **each argument value on its own** (80 chars)
   before cutting the whole rendering (200), so a call carrying a whole file still shows the file
   *name*; results and errors are folded the same way. `JLineTerminal.line` splits a multi-line string
   as a backstop for a caller that forgets. Only the console is cut — the model gets everything, and
   `ConsoleSession.rounds()` keeps the full arguments for the history note and `/calls`.
11. **The approval mode carries a glyph, and shift+tab switches it**: `ApprovalMode.symbol()` /
   `badge()` render `⏸ manual` and `⏵⏵ auto` on the status line and in `/mode`, the transport symbols
   the established terminal agents use for the same distinction; `ApprovalMode.next()` is the cycle
   the key walks. The binding is `AgentTerminal.onCycleMode(Runnable)`, which **defaults to declining**
   — only `JLineTerminal` overrides it, and the startup line advertises the key only when the bind
   succeeded. Two details are not obvious: it is bound **both** through terminfo
   (`InfoCmp.Capability.key_btab`) **and** to the literal `ESC [ Z`, because JLine's
   `windows-vtp.caps` declares no `key_btab` at all while the terminal in virtual-terminal input mode
   does send the sequence; and the shortcut fires **only while a line is being read**, so it switches
   the mode between turns — which is when it is decided anyway. The REPL therefore holds the two
   status numbers in an `AtomicLong`/`AtomicBoolean` rather than locals, so the widget (which runs
   inside the reader) can re-render the pinned row with what the last turn left behind.

12. **The input is framed into the pinned block, and the prompt stays there during a turn; typing
   stops the turn.** The frame is two halves that must be read together: the **top** rule is the first
   line of the reader's *prompt* (`rule() + newline + "> "`, rebuilt on every read because the window
   can be resized) — **no, and the second attempt was wrong too.** Both are recorded because the
   obvious fix is the one that fails. (1) Rule as the first line of a **two-line prompt**:
   `ERASE_LINE_ON_FINISH` erases exactly **one** line, so every Enter leaves the rule behind and
   holding Enter draws a column of them. (2) Rule as **ordinary output before each read**: nothing is
   left behind on Enter any more, but one rule now stays in the scrollback per turn and travels up
   with it. **There is no third option** — JLine's status region is below the prompt and never above
   it, so a rule above the input can only be part of the prompt (1) or part of the scrollback (2).
   The settled shape is therefore **one** rule, the first line of the status block, directly under the
   input line; the prompt is `"> "`, one line, and `AgentTerminal.readLine`'s `prompt` argument is
   consequently **ignored** here.
   `ERASE_LINE_ON_FINISH` removes the input line on Enter and the reader thread echoes it above as
   `› text`, so the transcript keeps what was asked.

   **`ScreenUseCasesTest` + `ScreenTerminalHarness` are how the resize and `/cls` cases are checkable,
   and they are what every byte-level attempt before them could not do.** The harness subclasses JLine's
   `LineDisciplineTerminal` and puts its own `ScreenTerminal` (a real VT interpreter, public API in the
   shipped jar) behind it, so a test reads the **screen** — "the block is smeared across the output",
   "an escape sequence is printed as text", "the block is drawn twice" are indistinguishable from
   correct output in a byte stream, which is why two byte-level assertions in `JLineTerminalTest` had to
   be deleted, one of them green with the fix it was written for switched off. The cases are the reports:
   dragging wider and narrower (with nothing typed, with text in the input, one step at a time, with a
   two- and a three-row block, with the real block's glyphs), `/cls` and Ctrl-L (block intact, and the
   cursor and prompt on the row the block leaves for them), a window that over-reports its width, size
   events arriving on another thread, and a block refresh racing the reader. Several assert the **cursor
   row** rather than the content, which is what finally located two of the defects: content can look
   plausible while the cursor is rows away from where a region reserved from the bottom expects it.
   Three things are load-bearing. **(1) A reader runs in every test** — it runs for the whole session in
   the application and owns the resize signal, so a test without one is a state the application cannot
   be in. **(2) The block rebuild is invoked directly, not waited for**
   (`refreshBlockForCurrentSize`): sleeping for the 120 ms poll passed alone and failed in a full run,
   and a flaky test is worse than none — what that leaves uncovered is the polling thread itself, a loop
   that compares two sizes and calls that method. **(3) A rule shows as `q` when it went out through a
   *prompt*** (the DEC line-drawing set, which this screen renders literally) and as `─` through the
   status region, so both forms count as a rule and a row of `q` also says which path it took.
   **The JLine fixes are demonstrated red/green through it**, the library being just a property — and since
   there are **eight** of them, one of which adds a method, the wiring needs a paragraph of its own.
   `ScreenUseCasesTest` **skips itself** on a JLine that does not carry them, keyed on
   `Status.repaint()` (the fifth fix and the only one that is a new method, so its presence stands in for the
   whole set). That is CLAUDE.md's own rule applied late — "no project test may assert the fixed behaviour
   while the build depends on an unfixed release" — and it had been broken: the pom's `jline.version` is the
   **released** one, which is what CI builds against, and against it these cases fail. Measured: the whole
   module is green with `-Djline.version=4.4.6-atmosphere` (the screen cases run) and green with the
   released `4.4.6` as well, where they report as **skipped** rather than passing.
   **Whether the fixes are still needed is measurable rather than arguable**, because the gate honours
   `-Datmosphere.screen.tests.runAnyway=true`. Against the current console, 233 tests: released `4.4.6` **26
   red**; `statusfix4` (fixes 1–4) 8; `statusfix5` (+`repaint`) 4; `statusfix6` (+addressing every row) 1;
   `statusfix7` (+padding one column short) **0**; `statusfix8` 0. (Those version names are the history of the
   investigation; **what to build today is `4.4.6-atmosphere`**, the reviewed set — the recipe is at the end of
   the investigation document.) So they are needed, by a wide margin — and the
   **eighth is carried on the merits, not on a failing test**: `Status.resize` still erases rows above the bar
   without it, but this console wipes and redraws after a width change, so nothing here observes it. It stays
   because the rows above a bar are not the bar's to clear, an erase is unrecoverable where a scroll is not, and
   every consumer that does not wipe needs it; its guard lives where the defect does, in `StatusRepaintTest`.
   **Every change was then reverted on its own and measured**, which dropped one and corrected a mistake of mine:
   the `xenl` entry for `windows-vtp` is **no longer caught by anything** (addressing every row and padding one
   column short removed the dependence on the wrap), so it is out — an unmeasured change does not belong in a set
   meant for submission. The `doDisplay()` → `display.resize(size)` change was first reported as unnecessary too,
   wrongly: that revert had been made with a string replacement matching nothing, compiled into a second directory
   layered in front of the first, which does not reliably win. Compiled properly, two tests fail without it. The
   set is therefore **seven** changes in three files, each with a failing test in JLine's own style behind it; the
   full matrix and both traps are in the investigation document. The marker cannot tell
   the fifth build from the later ones (the sixth adds a *protected* method, the seventh changes only a padding
   width), so on an older patched jar one or two cases fail rather than skipping — stated in the test rather than worked around.
   **The gate is the whole class and there is no fixed list of affected cases, which is itself a finding.**
   Against the released library between six and thirteen of them fail, a *different set each run*: the fourth
   fix is a data race, and when its `ConcurrentModificationException` lands on the reader's signal thread it
   ends that thread, after which no size change is reported at all and whichever cases were still to run fail
   too. The skip is per test (`@BeforeEach`), not a `@BeforeAll` assumption, because a class-level one makes
   Surefire record the class as **zero tests** — which reads as "nothing here" instead of "skipped", the same
   trap that silently muted every model-backed test in this repository for months.
   **What proves the fixes themselves are JLine's own tests**, in the clone's own style and next to its
   others: `StatusRedisplayTest`, `StatusDelayedWrapTest`, `StatusConcurrencyTest`, `StatusRepaintTest`
   (4 cases: an update with unchanged lines leaves damage on screen, `redraw()` does too because it is the
   same diff, `repaint()` puts every reserved row back without scrolling anything, and a repaint before
   anything was shown is not an error) and `StatusWrongWidthTest` (3 cases on a screen wider than the width
   it reports: every reserved row keeps its own screen row, nothing is written beside the rule, and the rows
   above the block stay empty, and — the seventh fix — a reported width one column too LARGE does not wrap the
   bottom row). 103 of JLine's own tests are green with all eight fixes, `DisplayTest` and `ScreenTerminalTest`
   included.


   **The fourth fix is a data race, and it is the one that explains the reports that survived the other
   three.** `refreshingTheBlockWhileTheReaderRedrawsNeverThrows` refreshes the block from three threads
   while the size changes, and fails with a `ConcurrentModificationException` whose stack names the
   defect: `AttributedString.join` iterating `Display.oldLines` from `Status.resize` (called by
   `LineReaderImpl.handleSignal`) while another thread is inside `Status.update` replacing that same
   `ArrayList`. Neither is synchronized and both are reachable from different threads *by design* — this
   console keeps its block current from its own thread four times a second. **Keystrokes do not provoke
   it; a size change does**, because the reader then resizes its display, resizes the region and
   redisplays. Raised on the input pump it ends that thread, after which no size change is ever reported
   again and the block keeps its width — which is the "rule wider than the window, wrapped" screen,
   reported as a separate problem. The fix is `synchronized` on `Status`'s nine public entry points.
   **And it is why the screen harness could not reproduce the reports it was built for**:
   `ScreenTerminalOutputStream.write` is synchronized, so it serialises the byte stream, while the
   unprotected state is JLine's own list — no stream lock reaches it. Two cases carry
   `@Disabled` as the record of a defect that is still open — a narrower drag loses the edit line, and a
   *three*-row block ends a wide drag with the rule three columns short, the state row shifted one
   column and the prompt on two rows, while a two-row block comes out clean.

   **`JLineTerminalTest` is how any of this is checkable**: `JLineTerminal.over(Terminal, …)` takes a
   terminal built over two streams, which renders exactly like a TTY, so the screen can be asserted on
   the emitted bytes. Two things that cost an hour each and are not guessable: the test terminal needs
   **`stdoutEncoding`** as well as `encoding`, or every `─` arrives as `?`; and a box character is
   written as UTF-8 from `printAbove` but as the **DEC line-drawing set** (`ESC(0` + `q`s + `ESC(B`)
   inside a *prompt*, so a counter that looks only for `─` passes against the exact bug it was
   written for — verified by putting the two-line prompt back and watching the test go from 1 rule to 5.

   **Escape sequences drawn as text** (`[?1h` above the prompt, then a `1H` inside the rule) were two
   further defects of the same family, and the second is the one that eventually **destroyed the
   block**. First: `line()` wrote straight to the terminal whenever the reader was not inside
   `readLine`, which is exactly when the next read emits its init sequence — once the reader thread
   exists, **everything** now goes through `printAbove`. Second: three threads write to this terminal
   as a matter of course — the turn (Atmosphere's thread) prints tool lines, the console thread
   refreshes the block four times a second, and with an in-process model **llama.cpp logs to stderr**,
   which is the same console and goes around JLine entirely. The first two are serialised by a
   `writing` lock held across `line()` and `status()`; the third is fixed by
   `LocalAgent.captureNativeLog`, which routes the native log through `LlamaModel.setLogger`
   (the callback sink `patches/0014` added) into `terminal.line`, so it scrolls in above the prompt
   like any other output instead of scrolling lines JLine never sees. **Honest limit:** the lock is
   reasoned, not test-covered. Two attempts to pin it are recorded in the history of
   `JLineTerminalTest` and both passed with the lock removed — even one that sliced every
   `OutputStream.write` in half — because `PrintWriter` already makes a single call atomic and the
   interleaving happens *between* calls, inside JLine. A test that is green either way is worse than
   none, so it was deleted rather than kept.

   **`/cls` clears by SCROLLING, not by erasing, and `/clear` also wipes the history.**
   `AgentTerminal.clearScreen()` defaults to doing nothing (a stream has no screen); `JLineTerminal`
   prints a window's worth of blank lines through `printAbove`. That is the whole implementation, and it
   replaces an erase-based one that took four attempts and produced a reported defect each time — which
   is why the reasoning is kept in full.

   **What a clear has to do here is two things at once**: leave the screen blank *and* leave the input on
   the row the pinned block leaves for it. Erasing does the first and undoes the second, because
   `clear_screen` puts the cursor home and the reader draws its prompt where the cursor is. The
   reports, in order: blank rows printed after the erase to push the input back down scrolled the
   *erased* lines back into view (`ESC[2J` clears the visible area and leaves them in the scrollback);
   a `cursor_address` smuggled into `printAbove`'s argument corrupted its bookkeeping — it moves up,
   writes, and redraws the prompt below — and stranded a character above the prompt; and the erase on
   its own left the input at the top left ("nach /cls ist der cursor auch ganz oben und nicht unten").
   The screen tests then showed that last state is worse than it looks: the reader redraws its prompt
   as a **diff** against what it believes is on screen, an erase invalidates that belief, and the
   measured result is **no prompt on screen at all** — cursor on row 0, the block still pinned at the
   bottom, ten blank rows between them.

   **Scrolling has none of those problems because it is nothing but output.** Everything is pushed above
   the window, so the screen is blank and what was written stays reachable with the scrollbar — which
   erasing the scrollback (`ESC[3J`) would have broken anyway, and this console promises it. Nothing is
   erased, so nothing can be pulled back into view. The reader's bookkeeping stays right, because
   printing above the prompt is what `printAbove` is *for*. The block is never touched: it is pinned, so
   it needs neither `status.reset()` nor a rebuild — three steps of erase-era repair went away with the
   erase. And the cursor ends on its row **by construction**, since printing is what pushes it there —
   which is also the mechanism behind "ein paar Mal Enter und alles sitzt wieder", so a screen that has
   lost rows is repaired by a clear rather than left crooked.

   **Ctrl-L is the same command through another door, and this console now owns the binding.** JLine's
   keymap dispatches Ctrl-L by *name* to the widget registered under `LineReader.CLEAR_SCREEN`, and
   JLine's own widget wipes and redraws the line — i.e. it reproduced the identical defect, measured on
   an interpreted screen. Replacing the map entry re-points the key without touching the keymap. **It
   deliberately does not take the `writing` lock**: a widget runs on the reader's thread with the
   reader's own lock held, while `line()` takes `writing` first and the reader's lock second, so
   acquiring `writing` there inverts the order and hangs the session. What serialises it instead is the
   reader's lock, which every `printAbove` needs; a concurrent block refresh can still interleave, which
   is the exposure JLine's own Ctrl-L widget has today as well.

   **Where each half is pinned.** `ScreenUseCasesTest` asserts the part that is only visible on a screen
   — after `/cls`, and again after Ctrl-L, the cursor is on `rows - 1 - blockRows`, the prompt is on that
   row, and every row above it is blank (checked one row at a time, so a failure names the row). All four
   were red before the change, with the cursor on row 0 and no prompt anywhere.
   `JLineTerminalTest.aClearScrollsAWindowAndErasesNothing` pins the pair a pipe *can* see: a window's
   worth of line feeds, and no `ESC[2J` at all. That test changed sides — it used to assert the exact
   opposite ("a clear must not scroll") and was right for as long as clearing meant erasing; both sides
   are recorded in it. Two byte-level assertions written for the erase (`erase display reached the
   screen`, `the block is drawn again after the wipe`) were **deleted** rather than adapted: what they
   described no longer happens, and the behaviour they were reaching for is asserted on the screen.

   **The screen is scrolled to the bottom once, before the first prompt** (`scrollToBottom`). The
   reader draws its prompt at the cursor, i.e. after the last line printed, while only the status
   block is pinned to the window — so on a half-empty screen the input floats in the middle with the
   block far below it, and they only meet once output has scrolled the cursor down by itself. That is
   why it looked right after a few turns and like an ordinary prompt at the start. Emitting
   `rows - 1` newlines once makes it the state from the first prompt on; from then on every printed
   line scrolls and the cursor stays on the last row. The cost is a screenful of blank lines above the
   session, which is what a program that wants its input at the bottom *without* taking over the
   screen has to pay.

   **The repeated `> Hallo` is a JLine defect, reproduced and fixed upstream-side.** Typing without
   Enter and then dragging the window showed the prompt and buffer a dozen times side by side. It is
   **one copy per size event** — a drag reports a new size per step — and an earlier reading of this
   ("the repeats are written while typing and widening only reveals them, so the count follows
   keystrokes") was an inference from a number rather than a measurement, and wrong. A three-row
   `Status` is equally necessary: the same drag without one is clean, and typing without a resize is
   clean too. Ruled out along the way: project code, JLine 4.4.6, `nativeSignals(false)`, a missing
   terminfo capability, Windows itself (it reproduces on a virtual `xterm`), and the `lastStatusSize`
   guard in `redisplay()` that was the documented hypothesis before the resize path was measured. The
   cause is `handleSignal(WINCH)`'s status branch — see the next paragraph. Full record, including the
   two measurements that were *inconclusive rather than negative*, in
   [`docs/upstream-investigation-jline-status-windows-redraw.md`](../docs/upstream-investigation-jline-status-windows-redraw.md).
   **Nothing here can honestly fix it**; `--plain` pins nothing and is unaffected.

   **Every output line is folded by this console, so the CONSOLE never wraps one** (`JLineTerminal.fold`, to
   one column less than the window). This is the fix for the whole family of drag artefacts and the reasoning
   is the load-bearing part: a line the console wrapped is **one** logical line spanning two screen rows, and
   Windows **joins such lines again when the window is widened**. The text above then needs fewer rows and
   everything below moves **up** — including the block rows last rendered, which end up above the pinned region
   where nothing ever writes again. One leftover per drag step, which is the reported staircase of rules
   climbing "von unten rechts nach oben links"; narrowing does it in reverse and walks the input upwards.
   **No program can observe a reflow or prevent one — but it can deny it a target:** a line that was never
   soft-wrapped has nothing to join.
   **The harness could not reproduce this, and that is why every grow test was green while the console was
   not:** `ScreenTerminal` pulls scrollback down when it grows and **does not reflow at all**. The property is
   testable, though, and that is what the two screen cases assert — after printing a long line, **no row on
   screen reaches the last column** (red before: rows 8 and 9 were full width). Five unit tests pin the fold:
   a short line is untouched, no piece exceeds the width, a double-width glyph is never split (a piece may come
   out a column short instead), ANSI a caller put in survives and does not count towards the width, and
   umlauts stay whole. Folding counts **screen columns** via `AttributedString.fromAnsi`, never characters —
   an icon is one character and two columns, which has cost this class a defect before.
   **The price is stated rather than hidden:** text keeps the line breaks it was printed with, so widening no
   longer re-flows the conversation. That is the same trade this console already makes by rendering
   append-only.

   **The pinned block is rebuilt whenever the window changes size, from a poll** (`startWatchingSize`,
   every 120 ms). What JLine holds are the rows it was handed, so a rule built for a 113-column window
   stays 113 columns wide: on a resize the row is padded with spaces or cut with an ellipsis, never
   re-made, and the next row then continues on the same screen line — the three rows run together with
   growing gaps, which is what was reported three times. Only the caller knows a rule is meant to span
   the window. **Both halves are needed and the second is easy to miss:** `status.resize(size)` first,
   or the pinned region cuts the rebuilt rule straight back to the old width. **Measured, not
   reasoned:** a probe on the real console reproduced the report when it left the block alone and
   rendered cleanly when it rebuilt all three rows per size event — after four harness theories had
   been measured and discarded (buffer-vs-window width, reflow by joining the rows, a
   wide→narrow→wide drag, an accumulating cursor drift), and with that probe recording window and
   buffer at identical widths throughout and one size event per ~125 ms for a single drag.
   **It carries no test, and a written one was deleted rather than kept:** the stream-backed harness
   has no screen model, so all it can observe after a size change is that JLine re-emits the rule at
   its *old* width without the poll and emits nothing with it — neither says the rule was redrawn at
   the new width, and the first assertion built on that passed with the poll disabled. Green either
   way is worse than none (the same call already made for two write-lock tests). The evidence is the
   probe, on the console where it happens. **The poll DOES re-assert the region
   (`status.resize(size)`), and that line was once removed for a measured reason before coming back for
   another:** it writes to the terminal directly, so without `Status`'s methods being synchronized it landed
   inside what the reader was drawing for the same size change and printed `36;1H` as text with the block
   doubled. That race is the fourth JLine fix carried here, so this line depends on that fix and must not be
   kept without it — `refreshingTheBlockWhileTheReaderRedrawsNeverThrows` is what holds the pair together.

   **The pinned region ADDRESSES each of its rows, which is the sixth JLine fix and the one that explains
   the screens full of rule fragments.** One measurement settled a long chase: a status update emits
   `ESC[8;1H` and then the rows back to back, padded to the reported width — **one cursor address for the
   whole bar**. The second row begins on a new screen row only because writing the last column of the first
   made the terminal wrap. A screen that is wider than the reported width (a window mid-drag, or one whose
   terminal has reflowed its buffer) therefore never wraps, and **every reserved row lands on one screen row,
   side by side**. And because the bar is reserved from the *bottom*, what the collapse pushes past the window
   lands in the **output area above it**, where nothing writes again — one fragment per drag, which is why it
   accumulated and why only `/cls` (which scrolls) cleaned it up.
   `Display.addressesEveryRow()` (false by default, so no other display changes) switches off the
   pending-wrap compensation, and `Status.MovingCursorDisplay` turns every cursor move into an absolute
   address. **Two narrower variants were measured and are wrong**, both recorded in the code: addressing rows
   at the top of the update loop makes even an unchanged update save and restore the cursor, which interleaves
   with the reader's writes and printed the typed text **one character per screen row** (deterministic, 2/2);
   addressing only row starts leaves a row that shares a prefix unaddressed, so the row above's pending wrap
   is never finished and the state row came out shifted one column, `" state]"` (deterministic, 3/3).
   **And it shipped with an off-by-one that JLine's own tests structurally could not see:** `Display` counts
   positions in `columns + 1` per row (`columns1`) throughout, the address divided by `columns`, so row *N*
   landed at column *N* — rule at 0, activity row at 1, state row at 2. Every existing test compares
   **trimmed** rows, which is exactly what hides it; it took the interpreted screen here to catch it, and
   `StatusWrongWidthTest.everyRowStartsAtColumnZero` now asserts on the raw row.

   **The prompt is never addressed at all, and that is the row that is still lost on a resize.** Bisected on
   the interpreted screen with the real block: after `/cls` the cursor is right (16 of 16), after **JLine's
   WINCH handling alone** it is wrong (19 of 20), and neither this console's row rebuild nor its repaint
   changes it either way. The byte stream says why — the status region clears its band, re-establishes the
   scroll region, restores the cursor, the block is addressed row by row, and then the prompt is written as a
   bare `>` **wherever the cursor happens to be**. Nothing relates it to the region below. A growing window
   moves the screen's content down by as many rows as it has scrollback to pull from, which need not be the
   number of rows added, so the prompt ends up a row or two above the rule.
   **Not patched, and the reason is a contract rather than cowardice:** a prompt directly above the pinned
   region is what *this* application wants (it scrolls to the bottom once at startup), while JLine promises
   only "the prompt is drawn at the cursor", which for a half-empty screen is correct. "Move the cursor to the
   bottom of the scroll region on a resize" would be right here and wrong there.
   `ScreenUseCasesTest.aGrowingWindowLeavesThePromptOnTheRowTheBlockLeavesForIt` carries the reproduction with
   the bisection in its `@Disabled` text; `/cls` repairs it in one keystroke, which the same measurement
   confirms.

   **The startup line names the terminal and the JLine patch level** (`LocalAgent.describeTerminal`), because
   its absence cost two rounds of testing: a screenshot from a console says nothing about which library
   produced it, and twice a defect was chased whose fix the jar in use did not contain. Probed by **method**
   (`Status.repaint` for the fifth fix, `Display.addressesEveryRow` for the sixth), never by version string —
   the patched builds overlay classes into the released jar and keep its manifest version, so the string
   cannot tell them apart.

   **A screen model that REFLOWS, because the one that does not reported every drag case green**
   (`ReflowingScreenHarness` + `ReflowResizeTest`, and `ReflowProbe` (in the agent's test tree, `…/atmosphere/probe/`) for the real console). JLine's
   `ScreenTerminal` adjusts its buffer on a resize and **never reflows**; a real console does, and that single
   missing behaviour is why several rounds of reproduction failed. The harness adds it and nothing else: it does
   not parse VT (JLine's screen keeps doing that), it reads the screen on a width change, rebuilds the logical
   lines, re-wraps them and writes the result back **behind JLine's back** — the honest channel, since a console's
   reflow changes the screen without telling the program. Both of its rules are explicit so they can be checked:
   a row **continues** into the next when its last cell is not blank (the terminal's own rule in practice), and
   which edge keeps its content when joining frees rows is a **parameter** (`Anchor.BOTTOM`/`TOP`), not a guess.
   `ReflowingScreenHarnessTest` puts the harness itself under test. **Its limit is stated:** the cursor is not
   reflowed with the content, so it is evidence about what is ON the screen after a resize, not about where the
   cursor lands — those cases stay with `ScreenTerminalHarness`.
   **The insight that came with it:** folding makes nothing soft-wrapped *at the width it was printed at*, so
   making the window NARROWER turns those same lines into wrapped ones and the next widening joins them. That is
   why enlarging alone looked fine while alternating "zerhackt alles" — the shrink manufactures what the widening
   then moves everything with.

   **The eighth JLine fix: `Status.resize` must not erase the rows above the bar.** The first thing the emulator
   showed was not a reflow but an **erase** — after a shrink the conversation was simply gone. Reproduced without
   any reflow at all, on the ordinary harness: three answers on a 100-column window, halve the width, and **two
   are erased**. `Status.resize` clears a band when the geometry changes and pulled its start upwards by
   `(ceil(oldColumns / columns) - 1) * statusLines` rows "to account for wrapped status lines" — six rows above a
   three-row bar in a halved window, straight through the application's output. Since the seventh fix a status row
   is padded one column short and **cannot wrap**, so there are no extra rows to account for. Red/green both
   sides: `StatusRepaintTest.makingTheWindowNarrowerDoesNotEraseWhatIsAboveTheBar` fails with the compensation
   restored and passes without it; `makingTheWindowNARROWERmustNotERASETheConversation` is red against
   `statusfix7` and green against `statusfix8`. **This is the answer to "nach dem kleiner ziehen sehe ich es nicht
   mehr":** the output was never scrolled away, it was erased by the bar's own housekeeping.

   **A status row must not write the last column, which is the seventh JLine fix.** `Status.update` padded every
   row to the **full** reported width. A row padded to a width the screen does not have **wraps**: it occupies
   two screen rows where the bar reserves one, so the bar's *last* row wraps past the bottom of the screen and
   **scrolls** it — the bar moves up and leaves a copy of itself above, one per bad render. That is the reported
   "ganz viel kleiner / größer abwechselnd zerhackt alles", a screen carrying several complete bars at different
   widths. The `…` some of those rules ended in is JLine's own marker for a row wider than the region, which is
   what pointed at it.
   **Padding one column short costs nothing** now that every row is addressed rather than reached by wrapping
   (the sixth fix), and it additionally stops the terminal from marking the row as **wrapped**, which is what let
   a reflow join it to its neighbour. Measured with a screen reporting one column more than it has — the lag a
   dragged console really shows: the state row came out as `" ▤ X:/tmp/…"`, its `[` eaten by the wrap; with eight
   columns over, eight characters. Both are the same thing the earlier `"state]"` reports were.
   **Our own half of that race was measured and is already closed:** a harness whose `getSize()` returns a
   different width on every call (a drag that stops) produces no ellipsis and a whole state row, because this
   console reads the size **once** per block build and hands that one read to both the region and the rows. The
   case is kept as the guard for it.
   **What the harness cannot show is the stack itself** — `ScreenTerminal` does not reflow, and JLine's diff skips
   a render whose content has not changed, so only the first bad render damages it. What it does show, a bottom
   row eaten by a wrap, is the same mechanism one render at a time.

   **The redraw after the size SETTLES is a repaint, not a diff** (`repaintBlockFromScratch`, 400 ms after
   the last size event). This is the fix for "kleiner ziehen sah gut aus, größer macht noch Probleme", and it
   rests on a mechanism worth knowing: JLine pads every region row to the width the terminal **reports** and
   writes the rows one after another, relying on the terminal wrapping at the right margin to start the next.
   A screen that is **wider** than the reported width therefore never wraps, and the next row continues on
   the same screen line — a rule with the activity row cut short beside it, the next rule where that left
   off, and the *last* row still perfectly correct (the region addresses its first row and then writes on, so
   only the rows in between collapse). Reproduced deterministically with a harness that reports eight columns
   fewer than its screen has.
   **Two causes reach that state and only one recovers by itself.** A console reporting a width it has not
   applied does it briefly, and the existing rebuild repairs it (measured, including a window that also grows
   a row taller). **Windows also reflows its screen buffer when the window is widened**, joining rows it had
   marked as wrapped — which is every region row, since each is padded to the last column — and JLine is
   never told. Its `Display` still matches what it wrote, so every later update computes an **empty diff and
   emits nothing** and the joined rows stay for the rest of the session. That is the half no earlier theory
   explained: the artefact *persisting* while every redraw runs, and repairing itself when Enter is held
   (printing does not go through the region's diff at all).
   **The repaint is two passes:** the same number of rows, blank, then the real rows — the first makes the
   second a real write and erases the rows on the way. **`Status.reset()` is the call that looks right and is
   wrong:** it forgets the scroll region too, so the next update believes it must grow the region and scrolls
   to make room — the stale rows were pushed *up* rather than cleared and the block stood on screen **twice**,
   four rows apart (measured on the interpreted screen, the only place that is visible). An empty block has
   the same problem for the same reason: it changes the region's height. Only the settle redraw repaints; a
   drag reports a size every ~125 ms and repainting on each is bytes spent against a screen about to change
   again.
   **What it cannot reach:** the prompt has a display of its own with the same diff and no `reset()` a caller
   can call, so a reflow that damages the prompt's row is still beyond repair from here —
   `thePromptItselfStaysVisibleAfterEnlarging` records that. `/cls` and Ctrl-L do repair it, because they
   only print.

   **A settled WIDTH change wipes the screen, in EITHER direction; a height change does not.** This is the last of the drag defects and
   the only one that could not be prevented, only removed. A console **reflows** when the width changes — measured
   on the reporter's, with a number rather than a screenshot, because a pasted screen cannot answer it (Windows
   Terminal copies the scrollback and rejoins wrapped runs, so wrapping is invisible in a paste): a 32-row window
   widened from 86 to 111 columns left the cursor on **row 28 of 32** where 31 is the last, exactly the three rows
   that joining three wrapped lines freed; narrowed to 72 it stayed on 31. So the **top** keeps its content and
   everything below moves **up** — and the block's rows are ordinary screen rows, so they are carried out of the
   region while the next render draws a fresh block at the bottom. The carried-up copy then sits above the region
   where nothing writes again: one per size event, one every ~125 ms during a drag, which is "beim größer ziehen
   wieder hunderte male die Linie".
   **Both halves of "cannot be prevented" were measured.** Taking the block off the screen at the first event of a
   drag fixes an *alternating* drag but not a plain shrink-then-widen: the console reflows immediately while this
   console learns of the size up to 120 ms later, so the block is unavoidably on screen then. Nothing can find the
   copy afterwards either — a caller cannot read the screen. Scrolling removes it, which is what `/cls` does and
   why that command was always the repair. The trade was the user's call: the visible conversation scrolls out of
   view on a width drag — and is then **printed again**, which is the part that makes the wipe affordable.
   Every line handed to `line()` is remembered **before** folding (a bounded ring of the last 300), and after the
   wipe as many of the most recent as fit above the block are printed again, oldest first, counted in **screen
   rows** rather than lines because folding can turn one line into several. Storing the text rather than the
   drawing buys what a reader expects from a window they just made bigger: the conversation **re-flows** — a line
   that needed two rows in a narrow window takes one in a wide one
   (`whatIsPrintedAgainIsFoldedForTheWidthTheWindowHasNOW` measures both states). It was wiped without this for
   one round and the report was a single sentence: "allerdings sehe ich den Verlauf nicht mehr". A height change
   needs none of this (no width change, nothing re-wraps) and keeps the conversation where it is, with the prompt
   printed back to its row instead.
   **The redraw only prints; it must not remember.** `line()` does both, and routing the redraw through it put the
   whole visible conversation into the ring a second time on every wipe — one turn ended up on screen four times
   ("der Text erscheint zwar wieder, aber mehrmals"). `remember` and `print` are separate for that reason. It also
   takes **two** width changes to see it: after the first wipe the ring holds the line twice but only one copy has
   been printed, which is why the case that shipped with the redraw was green while the console was not.
   **It was narrowed to "only when the width grew" once, on reasoning rather than evidence, and the next report
   came from the other direction:** narrowing does not join lines, it **splits** them, and the bar's own rows —
   built for the old width — no longer fit and are re-wrapped across several screen rows, so the bar needs more
   rows than the region reserves and everything above it is pushed up ("beim kleiner ziehen wandert es nach oben
   mit ganz vielen Zeilen"). Both directions leave rows behind that only scrolling removes. Note what the wipe is
   **not**: the eighth JLine fix stops the library from *erasing* those rows, which is unrecoverable, while
   scrolling them into the scrollback is not — that fix still matters for every consumer that does not wipe.

   **A height change moves the prompt off its row, and the settle prints it back**
   (`pushThePromptBackToItsRow`). Measured on the interpreted screen with the real three-row block: growing 20
   rows to 26 left the prompt on row 19 where 22 is right, shrinking 20 to 14 left it on row 13 where 10 is
   right — **half the height change in both directions**, and "half" is an artefact of that buffer state rather
   than a rule (the screen moves content by as many rows as it has scrollback to give or room to delete, which
   is what a console does). **A width change moves nothing at all**, which is folding paying off, so nothing is
   printed for one.
   **The growing half is fixed and the reason it works is the direction:** the drift cannot be computed, but it
   is bounded by the height change, and printing moves the cursor down one row per line until it reaches its row
   and then merely scrolls — so printing that many lines lands it correctly without knowing where it was. It is
   the same repair `/cls` performs with a whole window's worth, which is why that command always worked.
   Measured 19 → 22; `aGrowingWindowLeavesThePromptOnTheRowTheBlockLeavesForIt` was the open reproduction and is
   now green twice in a row. The price is bounded by the change: up to that many blank rows enter the
   conversation and up to that many lines of it scroll out of view.
   **The shrinking half stays open**, because there the prompt must move *up*: it lands inside the pinned band
   and the block draws over it ("nach dem kleiner ziehen sehe ich es nicht mehr"). Nothing a caller can emit
   moves the cursor up without breaking the reader's bookkeeping — `cursor_address` inside `printAbove` was tried
   twice and stranded characters above the prompt both times.
   `aWindowThatGetsSHORTERLeavesThePromptOnTheRowTheBlockLeavesForIt` carries it, `/cls` repairs it.
   **Measuring the cursor was ruled out, not overlooked:** a cursor-position report is a round trip through the
   terminal's input, which the reader owns all session, so it would race the keyboard and could swallow a
   keystroke or leave `[24;1R` in the input line — the exact class of defect this whole investigation has been
   about.

   **Do not add a `WINCH` handler, and the reason is measured.** A resize drawing a row of
   `> > > > >` across the screen looks like the pinned region not being told about the new size, so a
   `Signal.WINCH` handler that resized and re-rendered it was added — and the user reported it
   **worse**, not better. `LineReaderImpl.handleSignal(WINCH)` already calls `Status.resize(Size)`,
   and the reader installs its own handler for as long as it is reading, which is the whole session;
   ours therefore either never ran or ran *in addition*, putting a second writer on the terminal from
   the signal thread at the exact moment the reader was redrawing. A probe driving a real
   `terminal.raise(WINCH)` against a **pipe-backed** terminal shows JLine doing it correctly on its
   own: scroll region reset, the rule re-cut to the new width, **one** prompt — but that probe was
   measuring the wrong thing, because a pipe has no screen to redraw onto. Against JLine's own
   `VirtualTerminal` (a real VT interpreter over a virtual screen) the same drag reproduces the report
   exactly, on `xterm` and `windows-vtp` alike, and the cause is a **one-line JLine defect**:
   `handleSignal(WINCH)`'s status branch calls `doDisplay()`, which replaces the `Display` with a fresh
   one that believes the screen is blank, so the following `redisplay()` paints the prompt as new
   content once per size event instead of as a diff (`display.resize(size)` is the fix, verified with
   three new tests plus JLine's 72 existing ones). Still nothing to fix *here*, and no project test may
   assert the fixed behaviour while the build depends on an unfixed release.
   `fit()` measuring in **screen columns** (`AttributedString.columnLength`) rather than characters is
   ours and does matter: an icon is one character and two columns, and a row wider than the window
   wraps onto a second screen line, which the reserved region cannot survive.

   **Two things tried and removed, both because measurement said they did nothing.** (1) Skipping a
   status write when the block is unchanged — removing the guard again left the emitted bytes
   identical, because JLine already skips an unchanged block. (2) Re-cutting the rows when the window
   width changed — `Status.resize()` re-cuts the rows it holds itself. Neither was kept with a comment
   claiming a benefit it does not have. The stray `?1h` consequently still has **no established
   cause**: the lock covers our writes, the reader's own are inside JLine.

   **Restoring the block after a wipe took three steps — and the whole repair went away with the wipe.**
   It is kept as a record because it is the same mechanism that eventually retired the wipe itself. The
   steps were `status.reset()`, then `status.update(List.of())`, then rendering it again from `requested`
   (the text the caller gave, not the rendered rows); with only the first two, `Status` drew the
   difference it computed against a belief the wipe had invalidated — observed as a single character
   emitted where a whole block was missing. The reader's own prompt has exactly that problem and no
   equivalent repair, which is why an erase left no prompt on screen at all. A clear that only scrolls
   invalidates nothing, so the block is simply left alone.

   **The turn after an interrupted one is pinned end to end** (`InterruptedTurnTest`): with a scripted
   backend behind the real `OpenAiCompatServer`, a turn is cut short by pending input and the next one
   is then driven through — it reaches the server, carries the interrupted question in its history,
   and answers. Reported as "it does not carry on by itself"; the mechanism works, so the cause of
   that report is elsewhere and is **not** claimed to be fixed. Two things the writing of it settled:
   a turn that finishes inside one activity tick is never even looked at for interruption (correct —
   there is nothing to cut short), which is why the scripted backend has to be made slow or the test
   proves nothing; and a second line typed during the replacement turn stops that one too, which is
   the design and not a defect, but looks from the outside exactly like a turn that never started.

   **A blank line must not count as pending input.** `hasPendingInput()` ignores blank lines but leaves
   them queued: counting them meant that holding Enter cancelled one turn per keystroke and produced
   nothing, while dropping them would break the approval prompt, where an empty answer means yes.
   `DISABLE_EVENT_EXPANSION` is set in the same builder because the reader's default treats `!` as a
   shell history expansion, which silently rewrites a request like `git commit -m "fixed!"`.
   **What cannot be done, asked and answered:** keep the block visible while the *user* scrolls the
   terminal's scrollback. That needs the alternate screen buffer, i.e. a full-screen application, which
   would give up the scrollback and the append-only property the whole console design rests on.

   **The prompt stays at the bottom during a turn, and typing stops the turn.** One thread inside
   `JLineTerminal` (`startReading`) sits in `readLine` for the whole session and fills a queue;
   **every** read in that class is served from it, because a terminal has one keyboard and two threads
   reading it take turns at random. That is why `readKey` no longer reads a single key in raw mode: the
   question is printed above the prompt and answered in the same input line (`y` + Enter). End of input
   cannot be a queue value, so a sentinel is queued and **put back on every take** — otherwise the
   second reader after Ctrl-D would see "nothing typed yet" instead of "no more input".
   `AgentTerminal.hasPendingInput()` is the peek the turn loop peeks with; it deliberately does **not**
   consume, so the line the interruption was triggered by is still there for the next `readLine` and
   becomes the next message. The stop itself is `AgentRunner.start(...)` →
   `runtime.executeWithHandle(...)`, whose handle closes the in-flight SSE stream (Atmosphere's own
   "D-6 built-in hard-cancel"); that also replaced `turn()`'s hand-rolled worker thread, since
   `executeWithHandle` dispatches on a virtual thread and returns at once. `awaitWithActivity` returns a
   three-valued `TurnEnd` rather than a boolean, because *interrupted* must not be reported as the
   *timed out* error the old `false` produced. **Order in that loop is load-bearing and a test pins the
   behaviour**: the `activity.isPaused()` check comes first, so while an approval question is open a
   typed line is its answer and not an interruption. **What this is not:** Claude Code injects a
   mid-turn message into the running loop; `AgentExecutionContext` is a record whose request is built
   once from `message()` + `history()`, with nothing to append to, so stop-and-resend is the achievable
   equivalent — and it acts immediately instead of waiting out a tool loop.

**Three front ends on one `AgentSession`: console, `--web`, `--acp`.** Everything that is not
presentation lives in `AgentSession` — history, the slash commands (`submit` returns `CONTINUE`/`EXIT`),
`/compact`, `/loop`, `/retry`, the transcript, the approval mode, cancellation (`cancel()`,
`stopRequested()`, `isBusy()`), and `runTurn` itself. A front end implements `SessionFrontend`: `line`,
a `renderer()` (a `StreamingSession` the turn forwards every event to), `await` (how it waits for a
turn — the console with its activity block, the others with `AgentSession.awaitQuietly`), `approvals()`,
`ask()` (a free question; `null` means the front end has none, and `/loop` then refuses rather than
guesses) and `commandOutput()` (a running command's lines). The console is `ConsoleFrontend` over the two
`AgentTerminal`s; `LocalAgent` is now only option parsing, the choice of front end and the console's
activity rendering. Three pieces are what keeps them from drifting apart:

- **`TurnRecorder` records, the front end renders.** The recorder is the `StreamingSession` handed to
  Atmosphere; it keeps text, tool rounds, token counts and the running tool, supplies the workspace
  filesystem via `injectables()`, and forwards everything downstream. `ConsoleSession` is now just a
  recorder with a `ConsoleRenderer`. So the tool note in the history, `/calls` and the transcript are
  identical whichever front end showed the turn.
- **`ModeGatedApprovalStrategy` wraps every front end's strategy.** Atmosphere asks the strategy per
  call; the wrapper reads the session's mode *at that moment*, so `/mode auto` in the browser, *Allow
  all* in an editor and Shift+Tab in the console all take effect for the very next call, and a front
  end with nobody to ask (`approvals()` = `null`) denies. `ConsoleApprovalStrategy` keeps its own AUTO
  check as well; it is redundant under the wrapper and harmless.
- **`AgentSessionTest` + `RecordingFrontend`** drive the session with no terminal, no browser and no
  editor, over the real `OpenAiCompatServer` and a scripted engine; the three front-end test classes
  then only need to prove their own mapping.

**`--web` (`WebServer`, `WebAgentEndpoint`, `WebFrontend`, `WebAccessGuard`, `WebConsole`).** Embedded
Jetty 12 (ee10 servlet + Jakarta WebSocket; `jetty-ee10-annotations` excluded, nothing here needs
annotation scanning) with an `AtmosphereServlet` whose annotation map is set **explicitly** to the one
`@AiEndpoint`. **The scanning trap, which cost an afternoon:** Atmosphere's `ClasspathScanner.preventOOM()`
turns classpath scanning **off** whenever `org.junit.jupiter.api.Assertions` is on the classpath, and in
servlet-initializer mode the explicit map is only processed *during* a scan — so the endpoint registered
fine from the jar and was missing in every test ("No AtmosphereHandler or WebSocketHandler installed").
The init params `ANNOTATION_PACKAGE="all"` and `SCAN_CLASSPATH="false"` make the explicit map the path in
both. One `AgentSession` per process, shared by every tab (`WebAgentEndpoint.SESSION`); a message while a
turn runs cancels it first, `/stop` only cancels, and `connection.complete()` is always sent so the
console's input unlocks. Approvals use Atmosphere's own `/__approval/<id>/approve|deny` protocol through
`ApprovalStrategy.virtualThread(registry)`, which the prebuilt console renders as Approve / Deny.
`WebAccessGuard` is a Jetty `Handler.Wrapper` in front of everything: loopback `Host` when bound to
loopback (DNS rebinding), `Origin` must match `Host`, `?token=` exchanged for an `HttpOnly;
SameSite=Strict` cookie plus a redirect, else `Bearer` or cookie, else `401` — constant-time comparison.
`WebConsole` serves the console pages with a fresh CSP nonce per response and answers
`/api/console/info`. `WebServerTest` (11) drives it over the JDK `WebSocket` speaking the atmosphere.js
wire protocol (`AtmosphereTestClient`: `len|payload` frames, `X-atmo-protocol`); **create the client
object before `buildAsync`**, or the handshake message races the listener into a thrown exception and
`request(1)` is never called — a flaky test that looked like a server bug.

**`--acp` (`AcpServer`, `AcpFrontend`).** The ACP Java SDK (`com.agentclientprotocol:acp-core` +
`acp-json-jackson2`, 0.18.0) over `StdioAcpAgentTransport`. Its sync handlers run on the SDK's own pool,
so a blocking `session/prompt` does not block `session/cancel` or the answer to a permission request.
One `AgentSession` per ACP session, built for the editor's `cwd` (`AgentOptions.withWorkspace`); the
model endpoint is opened once per process. **stdout is the protocol**: `run()` keeps the real stdout for
the transport and points `System.out` at stderr, so a stray `println` cannot corrupt the stream (the
smoke asserts every stdout line is JSON). Mapping: text → `agent_message_chunk`; `ToolStart` →
`tool_call` (`call_N`, `ToolKind` from the tool name, `locations` resolved against the workspace, raw
arguments), `ToolResult`/`ToolError` → `tool_call_update` (result cut at 4000 chars), running command
output → `tool_call_update` IN_PROGRESS at most every 500 ms with the last 40 lines; approvals →
`session/request_permission` with allow / always / reject, where *always* switches the session to AUTO
and sends `current_mode_update`, and no answer (timeout, cancelled dialog, editor gone) is a deny; the
approval modes are ACP session modes `manual`/`auto`; the slash commands (minus `/exit`, `/cls`) go out as
`available_commands_update` 100 ms after `session/new` on a virtual thread, because an update that
arrives before the `session/new` response names a session the editor does not know yet.
`AcpServerTest` (9) deliberately drives it with **plain JSON** (`AcpTestClient`), not the SDK's client —
an SDK on both ends agrees with itself even where it disagrees with the protocol. Tear-down closes the
editor side first: that is how an ACP session ends, and closing the server first logs an
`InterruptedIOException` "Transport error" per test.

**Two defects fixed on the way (step 0), both pinned in `LocalAgentTest`:** the shell tool's live-output
sink captured the console *before* it existed, so every command's output came back as
`[output unavailable: NullPointerException]` (now an `AtomicReference` set once the terminal is up); and
`/clear` left the pending tool note and the message `/retry` would resend, so the first turn after a
clear still carried the old conversation.

**The default system prompt is general-purpose on purpose — do not narrow it back.** Every model-facing
text is a resource, not a Java literal: `src/main/resources/net/ladenthin/llama/atmosphere/` holds
`system-prompt.txt`, `system-prompt-shell.txt`, `system-prompt-no-shell.txt` and `run-command-tool.txt`
(each with a `.license` sidecar for REUSE), loaded by `LocalAgent.prompt(name)` with `{placeholder}`
substitution; `AgentOptionsTest.promptResourcesLoadAndEveryPlaceholderIsFilled` fails on a missing file or
an unfilled placeholder. `LocalAgent.systemPrompt`
and the `ShellTool` description describe `run_command` as running *any* command line through the named
shell (`ShellTool.shellName()`), not limited to the workspace, and tell the model to run a command rather
than explain one. The earlier wording ("careful *coding agent*", `run_command` "to build, test or inspect
the project" / "build, test, grep or list files") made Qwen3-4B refuse "list the docker images" — "my
tools are only for files" — with the tool registered and `docker` on `PATH`; a fresh single-turn run
refused too, so it was the prompt, not the chat history. Without `--allow-shell` the prompt says commands
are unavailable and names the flag, so the model does not invent its own limitation. Pinned by
`AgentOptionsTest.defaultSystemPromptIsGeneralPurposeAndAllowsAnyCommandWithTheShell` and
`shellToolDescriptionDoesNotNarrowItToTheProject`. `ShellToolTest` runs on every platform: each test
picks its command line with `ShellTool.isWindows()` — the same detection `ShellTool.run` uses to choose
`cmd.exe /c` over `sh -c` — so `ls`/`dir /b`, `sleep 30`/`ping -n 30 127.0.0.1 >nul`, and the truncation
test counts the platform's line separator. It used plain POSIX commands before and failed 4 of 5 on
Windows; never skip it per OS, give a new test both command forms instead.

**Version bump note.** The pom's own `<version>` must equal the reactor's (`check-natives.py` enforces
it); `versions:set` does not touch this standalone pom, so a release bumps it by hand, together with the
version in the two READMEs (`README.md` "Local coding agent" + the project's own README: the JBang
coordinates and the fat-jar filenames) — the same class as the `llama-langchain4j/README.md` snippet.
`llama.version` follows by itself (`${project.version}`); `-Dllama.version=<x>-SNAPSHOT` still runs it
against another core (the pom keeps the Sonatype snapshot repository for exactly that).
