<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT
-->
# LLM Service — requirements (current behavior)

The **specification of record** for what the **LLM Service** Android app (`android-llmservice/`)
does **today**. Because the app has **no unit tests** for its logic — the only automated check is
the single on-device `ChatFlowInstrumentedTest`, which exercises just the core load→type→send→stream
path — this file is the authoritative description of the app's behavior. Each requirement is written
to be individually testable, so tests can later be written against these IDs.

**Relationship to the other docs (keep all three in sync):**

- **`requirements.md`** (this file) — what the app **currently implements** (shipped behavior).
- **`TODO.md`** — the **roadmap** (not-yet-built features, t-shirt-sized). When a TODO item ships,
  flip it to ✅ in `TODO.md` **and** add/expand the matching requirement here.
- **`README.md`** — the human-facing overview. Its feature claims must match this file.

**Sync rule:** any change to app behavior (a new feature, a changed default, a removed control)
**must** update this file in the same commit — add a new `Rn.m` row, amend the affected row, or mark
it removed. Do not let a shipped behavior exist without a requirement here. IDs are **stable**: never
renumber; retire a requirement by marking it ~~struck through~~ with a note rather than reusing its ID.

**"Verified by" legend:** `instrumented` = covered by `ChatFlowInstrumentedTest`; `manual` = verified
by hand only (no automated test); `build` = enforced at build/resource-compile time.

---

## R1 — App identity & packaging

| ID | Requirement | Source | Verified by |
|---|---|---|---|
| R1.1 | The app label is **"LLM Service"** (not translated — same in every locale). | `values/strings.xml` `app_name`; `AndroidManifest.xml` | manual |
| R1.2 | applicationId / namespace / Kotlin package is **`net.ladenthin.android.llmservice`**. | `app/build.gradle.kts` | build |
| R1.3 | Minimum SDK is **28** (Android 9 Pie — the AAR floor); target/compile SDK **35**. | `app/build.gradle.kts` | build |
| R1.4 | Native ABIs shipped: **`arm64-v8a`** (phones) and **`x86_64`** (emulators / x86 Android). | `app/build.gradle.kts` | build |
| R1.5 | Depends on `net.ladenthin:llama-android` (AAR) + `net.ladenthin:llama-kotlin` (façade), version selectable via `-PjllamaVersion`. | `app/build.gradle.kts` | build |
| R1.6 | It is a standalone plain-Gradle/AGP build (NOT a Maven reactor module) and is **not** published to Maven Central. | repo layout | manual |
| R1.7 | The launcher icon is the **shard-`J`** of the java-llama.cpp logo (orange `#F65E00` on `#111111`), shipped as an **adaptive icon** (foreground + `@color/ic_launcher_background`) for API 26+ plus legacy `mipmap-*/ic_launcher.png` for older devices; wired via `android:icon`/`android:roundIcon`. Sourced from the shared `workspace` logo set. | `AndroidManifest.xml`; `res/mipmap-anydpi-v26/ic_launcher.xml`; `res/mipmap-*/ic_launcher*.png`; `res/values/ic_launcher_background.xml` | manual |

## R2 — Privacy & network posture

| ID | Requirement | Source | Verified by |
|---|---|---|---|
| R2.1 | The app declares **no `INTERNET` permission** and **no storage permission** — nothing leaves the device. | `AndroidManifest.xml` | build |
| R2.2 | `allowBackup=false` — conversation data is not included in system backups. | `AndroidManifest.xml` | build |
| R2.3 | All inference runs **on-device** (CPU); no request is ever made to a remote server by the app. | `ChatViewModel` (no network calls) | manual |
| R2.4 | A **"🔒 Offline · fully on-device"** badge is shown on the model-picker screen. | `MainActivity.OfflineBadge`; `badge_offline` | manual |
| R2.5 | **Transient working data is ephemeral:** the copied model (`current-model.gguf`), the copied vision projector (`current-mmproj.gguf`, see R12), and the cache dir are **wiped on every cold start** (`LlmServiceApp.onCreate`) — guaranteeing a fresh start regardless of how the app was last killed — and best-effort on finish (`MainActivity.onDestroy` when `isFinishing`). The **only** data that persists is the user's **explicitly saved** session (`session.json`, opt-in). | `LlmServiceApp`; `MainActivity.onDestroy` | manual |
| R2.6 | **Quit & clean up:** a ❌ button on the model-picker (main) screen opens a confirm dialog that, on confirm, **wipes the working data** (model copy + cache, keeping the saved session) and **closes the app** (`finishAndRemoveTask` — removed from Recents). | `MainActivity` (`quitButton`; `LlmServiceApp.clearWorkingData` + `finishAndRemoveTask`) | manual |

## R3 — Model selection & loading

| ID | Requirement | Source | Verified by |
|---|---|---|---|
| R3.1 | The user picks a GGUF via the **Storage Access Framework** (`ACTION_OPEN_DOCUMENT`, `*/*`) — no storage permission. | `MainActivity` picker | manual |
| R3.2 | A picked `content://` model is **copied into app-private `filesDir`** (`current-model.gguf` = `MODEL_COPY_NAME`) and loaded **by real path** (llama.cpp mmaps a real path, not a `content://` URI). The copy is **ephemeral** (wiped on cold start / close — see R2.5), so a model is re-picked after the app is fully closed. | `ChatViewModel.copyUriToInternal` | manual |
| R3.3 | The model loads **CPU-only** (`setGpuLayers(0)`) with context size **2048**, portable across every device. | `ChatViewModel.openModel` (`CONTEXT_SIZE`) | manual |
| R3.4 | While loading, a **LOADING** state is shown (spinner + "Loading model…"); on failure a localized load error is shown and state returns to NONE. | `ChatViewModel.ModelState`; `error_load_model` | manual |
| R3.5 | Loading a new model **closes** the previously loaded one (native memory is not GC-managed). | `ChatViewModel.openModel` | manual |
| R3.6 | Recommended real model for coherent replies: an instruct GGUF (default **Gemma 3 4B Instruct**); base/completion models still run. | `README.md` | manual |
| R3.7 | The model-picker screen shows a **device-readiness card**: free/total **RAM** (`ActivityManager.MemoryInfo`), free **storage** (`StatFs`), and **battery** level + charging (`BatteryManager`) — so the user can gauge whether a model fits before loading. Read once at composition. | `DeviceInfo`; `MainActivity.DeviceCard` | manual |
| R3.8 | **Unload model:** a ❌ button next to the model name (shown only when a model is loaded) frees the native model from memory, deletes the copied working model file, and returns to the "no model loaded" state. The **chat history is kept** (still saveable/copyable); a generation in flight is cancelled first. | `MainActivity` (`unloadButton`); `ChatViewModel.unloadModel` | manual |

## R4 — Chat & streaming

| ID | Requirement | Source | Verified by |
|---|---|---|---|
| R4.1 | The assistant reply **streams token-by-token** into the UI over the `llama-kotlin` `generateChatFlow` façade; native work runs on `Dispatchers.IO`. | `ChatViewModel.startGeneration` | instrumented |
| R4.2 | Messages render as **bubbles** (user vs assistant styling); the list **auto-scrolls** to the newest message. | `MainActivity.MessageBubble` / `Conversation` | manual |
| R4.3 | A **localized system prompt** is passed on every turn, nudging the model to answer in the user's language. | `system_prompt`; `ChatViewModel.send` | manual |
| R4.4 | `send` is a **no-op** when the input is blank, no model is loaded, or a generation is already in flight. | `ChatViewModel.send` | manual |
| R4.5 | An optional **chat-template override** (e.g. `chatml`) is supported for template-less GGUFs (used by the test hook; real instruct models carry their own template). It is applied at **model load** (`ModelParameters.setChatTemplate`) — per request it was silently discarded by llama.cpp's schema, so the hook proved nothing. | `ChatViewModel.openModel` | instrumented |
| R4.6 | A generation error keeps the partial reply and surfaces a localized generation error. | `ChatViewModel.startGeneration`; `error_generation` | manual |
| R4.7 | **Prompt shortcut chips** (localized `SuggestionChip`s) appear above the input when ready/idle and the input is empty; tapping one fills the draft with a quick-start prompt. | `MainActivity.Conversation` (`promptChip`); `chip_*` | manual |

## R5 — Generation settings (sampling)

| ID | Requirement | Source | Verified by |
|---|---|---|---|
| R5.1 | A **Settings sheet** (⚙️ in the top bar) exposes live sliders for **temperature, Top-K, Top-P, Min-P, repeat penalty, repeat range, max tokens**, plus **Reset defaults**. | `MainActivity.SettingsDialog`; `ChatViewModel.GenerationSettings` | manual |
| R5.2 | Defaults are **temperature 0.7, Top-K 40, Top-P 0.95, Min-P 0.05, repeat penalty 1.1, repeat range 64, max tokens 256**. | `ChatViewModel.GenerationSettings.DEFAULT` | manual |
| R5.3 | All seven knobs are forwarded verbatim to `InferenceParameters` on every generation. | `ChatViewModel.startGeneration` | manual |
| R5.4 | The **repeat penalty > 1 with a non-zero repeat range is required** — it is what prevents the degenerate repetition loop small on-device models fall into with a bare decode. | `ChatViewModel.startGeneration` (comment) | manual |
| R5.5 | Slider ranges: temperature `0–2`, Top-K `0–100`, Top-P/Min-P `0–1`, repeat penalty `1–2`, repeat range `0–256`, max tokens `16–1024`. | `MainActivity.SettingsDialog` | manual |
| R5.6 | The Settings sheet also has a **"Model" section** for the **load-time** params **CPU threads** (`1–16`, default 4) and **Context length** (`512–8192`, default 2048). These are **not** live — a **Reload model** button (enabled only when a model is loaded) reopens the current model with the new values. | `MainActivity.SettingsDialog`; `ChatViewModel.ModelConfig` / `reloadModel` | manual |

## R6 — Chat actions

| ID | Requirement | Source | Verified by |
|---|---|---|---|
| R6.1 | **Stop:** while a reply streams, the Send button is replaced by a **Stop** button that cancels the generation, **keeps the partial reply**, and shows **no** error (cancellation is not treated as a failure). | `MainActivity.Conversation`; `ChatViewModel.stopGeneration` | manual |
| R6.2 | **Copy:** the last assistant reply can be copied to the clipboard (with a "Copied" toast). | `MainActivity.Conversation` (`copyButton`) | manual |
| R6.3 | **Regenerate:** re-runs the last user turn — drops the trailing assistant reply and regenerates from the conversation up to that user turn (using the current sampling settings). | `MainActivity` (`regenerateButton`); `ChatViewModel.regenerate` | manual |
| R6.4 | Copy/Regenerate appear only when idle and the last message is a **non-empty assistant reply**. | `MainActivity.Conversation` | manual |
| R6.5 | **Clear chat:** 🗑 in the top bar (enabled only with messages present and not generating) clears the conversation after a **confirm dialog**. | `MainActivity` (`clearButton`); `ChatViewModel.clearChat` | manual |
| R6.6 | **Long-press any message bubble** copies that message's text to the clipboard (with a "Copied" toast) — complements the last-reply Copy button (R6.2). | `MainActivity.MessageBubble` (`combinedClickable`) | manual |

## R7 — In-app log

| ID | Requirement | Source | Verified by |
|---|---|---|---|
| R7.1 | An **always-visible one-line log strip** sits at the very bottom (🧾, small font, may wrap to 2 display lines) showing the newest log line, or an empty-state string. Under edge-to-edge (`enableEdgeToEdge`) its content is padded **above the system navigation bar** (`navigationBarsPadding`) so it is never covered by the back/home/recents bar. | `MainActivity.onCreate`; `MainActivity.LogStrip` | manual |
| R7.2 | Tapping the strip opens a **full-screen log viewer** with **Copy all**, **Save as…** (writes a `.txt` via SAF `CreateDocument`), and **Clear**; ✕ (top-right) closes it. | `MainActivity.LogDialog` | manual |
| R7.3 | The log is a **rolling buffer capped at 500 lines**, each entry prefixed with a local wall-clock time. | `ChatViewModel.log` (`MAX_LOG_LINES`) | manual |
| R7.4 | Logged events include: model loading/ready/load-failure, generation start (with the effective sampling knobs), reply-complete (char count), generation-stopped, generation-failure, regenerate, chat-cleared, settings-reset, and session save/load. | `ChatViewModel` (`log(...)` call sites) | manual |

## R8 — Session persistence (private, local)

| ID | Requirement | Source | Verified by |
|---|---|---|---|
| R8.1 | **Save** (💾) writes the conversation **and the model's on-device path** to a single JSON file in app-private storage (`filesDir/session.json`), readable only by this app. | `ChatViewModel.saveSession`; `SessionStore` | manual |
| R8.2 | **Load** (📂) restores the saved messages and, if the model file is still present and different from the current one, **re-opens the model**. | `ChatViewModel.loadSession` | manual |
| R8.3 | Save/Load are **opt-in** (nothing autosaves) and fully local (never uploaded). | `ChatViewModel` | manual |
| R8.4 | Save/Load/none outcomes are surfaced as localized one-shot **snackbar notices**. | `MainActivity` `noticeRes`; `toast_session_*` | manual |

## R9 — Internationalization

| ID | Requirement | Source | Verified by |
|---|---|---|---|
| R9.1 | Every user-facing string is a resource; translations ship for **13 languages** (en, de, es, fr, it, pt, ru, tr, ar, hi, zh-CN, ja, ko) plus a **System** (device-language) option. | `values/` + `values-*/strings.xml`; `Languages.kt` | build |
| R9.2 | A **flag-emoji dropdown** in the top bar switches language **in-app** via AndroidX per-app locales (`AppCompatDelegate.setApplicationLocales`), persisted across restarts and integrated with the Android 13+ per-app language setting. | `MainActivity.LanguageMenu`; `res/xml/locales_config.xml` | manual |
| R9.3 | The ViewModel **survives** the locale-change activity recreation, so the chat is not lost when switching language. | `MainActivity` (`by viewModels()`) | manual |
| R9.4 | `app_name` ("LLM Service") is the one string deliberately **not** translated. | `values*/strings.xml` | manual |
| R9.5 | The set of translated strings is **consistent across all locales** — every key in the default resources has a translation in each `values-*` file. | `strings.xml` files | build |

## R10 — UI shell & signing

| ID | Requirement | Source | Verified by |
|---|---|---|---|
| R10.1 | The app bar is **two rows**: row 1 is the **model-name title** (own full-width line, **horizontally scrollable** / draggable, single line, no auto-marquee) plus a **❌ unload button** at its right when a model is loaded (R3.8); row 2 holds **all action icons** (save / load / clear / settings / language) on their own line below it. | `MainActivity` top bar | manual |
| R10.2 | The release `signingConfig` reads an **upload keystore** from env vars / `-P` props, and **falls back to debug signing** when none is set (so forks/PRs/local builds stay green). **Upgrade caveat:** the debug fallback uses an ephemeral per-runner key, so debug-signed CI APKs are **not** mutually upgradeable (a signer change is rejected → uninstall required); stable in-place upgrades need the upload-key secrets. | `app/build.gradle.kts` | build |
| R10.3 | The build produces a release **`.aab`** (for Play) and an installable release **`.apk`** (sideload); the Maven Central GPG key cannot sign these — Android needs a Java keystore upload key. | `app/build.gradle.kts`; `README.md` | build |
| R10.4 | `versionCode` is **monotonic** (derived from `GITHUB_RUN_NUMBER` in CI, strictly increasing per run; `-PappVersionCode` overrides; local hand builds use `1`), so successive release APKs advertise a higher version and Android accepts the in-place upgrade (given a stable signer, R10.2). | `app/build.gradle.kts` | build |

## R11 — Testing hooks & automated coverage

| ID | Requirement | Source | Verified by |
|---|---|---|---|
| R11.1 | `MainActivity` reads optional `EXTRA_MODEL_PATH` / `EXTRA_CHAT_TEMPLATE` intent extras to preload a model by absolute path, bypassing the SAF picker — a **test hook** the shipping UI never sets. | `MainActivity` companion | instrumented |
| R11.2 | `ChatFlowInstrumentedTest` launches the app with a preloaded model, types a prompt, taps Send, and asserts a **non-empty streamed assistant reply**; it self-skips when the adb-pushed model is absent. | `androidTest/.../ChatFlowInstrumentedTest.kt` | instrumented |
| R11.3 | **Coverage gap:** there are currently **no unit tests** for `ChatViewModel`, `SessionStore`, `Languages`, or the settings/log/chat-action logic. This file is the spec of record until such tests exist. | — | — |

## R12 — Vision (mmproj) model + image attachment

| ID | Requirement | Source | Verified by |
|---|---|---|---|
| R12.1 | The model-picker screen has an **optional** "Add vision" button that picks a vision projector (**mmproj**) GGUF via SAF, independent of and in either order relative to the main model. Once set, the button shows the projector's file name and a ✕ to clear it. | `MainActivity.ChooseModelView` (`chooseMmprojButton`/`clearMmprojButton`); `ChatViewModel.loadMmprojFromUri`/`clearMmproj` | manual |
| R12.2 | The picked mmproj is **copied into app-private `filesDir`** (`current-mmproj.gguf` = `MMPROJ_COPY_NAME`, same real-path-not-URI reasoning as R3.2) and passed to `ModelParameters.setMmproj(...)` (+ `setDevices("none")` + `setMmprojOffload(false)`, mirroring the CPU-only config `MultimodalIntegrationTest` validates) the next time a model is loaded. | `ChatViewModel.openModel` | manual |
| R12.3 | Once a vision projector is loaded, a **📎 attach** button appears in the chat input row; tapping it picks an image (`image/*`) via SAF and stages it as a **pending attachment chip** (🖼️ filename + ✕ to remove) above the input. | `MainActivity.Conversation` (`attachImageButton`/`pendingImageChip`/`removeAttachmentButton`); `ChatViewModel.attachImage`/`clearPendingImage` | manual |
| R12.4 | **Send** accepts either non-blank text, a pending image, or both; a message with an image is built as a multimodal `ChatMessage` (`ContentPart.text` + `ContentPart.imageBytes`) instead of plain text. | `ChatViewModel.send`/`Message.toChatMessage` | manual |
| R12.5 | Attached images are **session-transient**: shown in the message bubble (🖼️ prefix) for the lifetime of the app process, but **not** persisted by `SessionStore` (save/load keeps text only) and not copied to disk (read into memory only). | `ChatViewModel.Message`; `SessionStore` | manual |
| R12.6 | **Unloading the model** (R3.8) also clears the selected vision projector (path + copied file) and any pending image attachment — picking a new model starts vision selection fresh. | `ChatViewModel.unloadModel` | manual |
| R12.7 | The vision projector copy is included in the same **privacy wipe** as the model copy: deleted on cold start (`LlmServiceApp.onCreate`) and on `clearMmproj`/`unloadModel`. | `LlmServiceApp.clearWorkingData` | manual |

---

## Known coverage gaps / follow-ups

- No unit tests for the ViewModel/session/settings/log logic (R11.3). Candidates: `SessionStore` round-trip
  (JSON save→load), `flagForActiveTags` mapping, `GenerationSettings` defaults/reset, `regenerate`
  history-trimming, log capping at `MAX_LOG_LINES`.
- Per-message copy (any bubble, not just the last reply) is a possible follow-up (see `TODO.md` §3).
- Roadmap items (model manager, server screen, download manager, multimodal, theming, onboarding)
  live in [`TODO.md`](TODO.md) — not part of the current requirements above.
