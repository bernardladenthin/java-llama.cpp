# CLAUDE.md — `android-llmservice/` (the "LLM Service" Android app)

Guidance for working in this directory. Claude Code loads it when files here are read; the
repository-wide rules are in [`../CLAUDE.md`](../CLAUDE.md), which keeps a short summary.


A shippable, **KISS fully-offline on-device chat app** consuming the `llama-android` AAR +
`llama-kotlin` façade. App label **"LLM Service"**, applicationId/namespace/package
**`net.ladenthin.android.llmservice`**. One Compose screen: pick a GGUF from the file system
(Storage Access Framework), stream a chat reply fully on-device, **tune sampling** via a Settings
sheet (⚙️ — temperature/Top-K/Top-P/Min-P/**repeat penalty**/repeat range/max tokens; the
repeat-penalty default is what stops the small-model repetition loop), **Stop/Copy/Regenerate/Clear**
the chat, watch an **in-app log** (bottom strip + full viewer with copy-all / save-as-`.txt` / clear),
switch UI language via a **flag picker (13 languages)**, and **Save/Load** the conversation to
**private local storage**. No `INTERNET`/storage permission — nothing leaves the device. Like `llama-android/` and
`.github/android-consumer-test/`, it is a **standalone plain-Gradle/AGP build, NOT a Maven
reactor module** (and NOT published to Maven Central — it is an app, not a library). It is the
"real app" counterpart to the headless `.github/android-consumer-test/` fixture (which only
compiles/loads the API); this one has a UI and is driven end-to-end. The name is intentionally
**generic/descriptive** (sidesteps trademark conflicts); the applicationId under the owner's
domain is the only string that must be globally unique.

Structure (mirrors the consumer-test's plumbing):
- **`settings.gradle.kts`** — `rootProject.name = "android-llmservice"`; pins AGP `9.4.0` + the
  Compose compiler plugin (`2.4.20`); `mavenLocal()` first so the freshly-built AAR + façade
  resolve there in CI (Maven Central for real users). Stay at `>= 9.2.1`: `9.2.1` (not `9.2.0`)
  first fixed a real R8 regression (`ClassNotFoundException` on `com.android.tools.r8.RecordTag`
  after upgrading Gradle to 9.x with AGP 9.2.0) that hits this project directly since
  `buildTypes.release` sets `isMinifyEnabled = true`; `9.4.0` carries that fix forward and the
  `.github/android-consumer-test` fixture is pinned the same way. AGP 9.4.x requires Gradle >= 9.6.0
  and JDK 17+; CI already runs JDK 21 everywhere (`.java-version`), so only the `gradle-version`
  pin on the jobs that build this project (and the `.github/android-consumer-test` fixture)
  needed bumping (currently `9.8.0`). AGP 9.0+ has **built-in Kotlin support** (a runtime dependency on
  Kotlin Gradle plugin 2.2.10+), so the standalone `org.jetbrains.kotlin.android` plugin is no longer
  applied — applying it now fails the build with "no longer required for Kotlin support since
  AGP 9.0" (`app/build.gradle.kts` line 7). The Compose compiler plugin still applies
  separately and its 2.4.20 pin exceeds AGP's 2.2.10 floor, so no other version changed.
- **`app/build.gradle.kts`** — `namespace`/`applicationId` `net.ladenthin.android.llmservice`,
  `minSdk 28` (AAR floor), `compileSdk 37` (raised from 35 — Compose/lifecycle/activity AAR
  metadata now requires it), `targetSdk 35`, Jetpack Compose, `androidx.appcompat` (only for
  the per-app language API), ABIs `arm64-v8a` + `x86_64`. Depends on `net.ladenthin:llama-android` +
  `net.ladenthin:llama-kotlin` at `-PjllamaVersion=<reactor version>` (defaults to the last release
  when built by hand). The release `signingConfig` reads an **upload keystore** (env vars
  `JLLAMA_UPLOAD_STORE_FILE` / `_STORE_PASSWORD` / `_KEY_ALIAS` / `_KEY_PASSWORD`, or the matching `-P`
  props) and **falls back to debug signing** when none is set, so forks/PRs/local builds stay green.
- **i18n:** every UI string is a resource; translations ship for **13 languages**
  (`values-{de,es,fr,it,pt,ru,tr,ar,hi,zh-rCN,ja,ko}/strings.xml` + default English), listed in
  `res/xml/locales_config.xml` and `Languages.kt`. The flag dropdown switches language in-app via
  `AppCompatDelegate.setApplicationLocales` (AppCompat `autoStoreLocales` service persists it;
  `MainActivity` extends `AppCompatActivity`, theme `Theme.LlmService` = AppCompat DayNight). The
  ViewModel survives the locale-change recreation, so the chat isn't lost. `app_name` ("LLM Service")
  is the one string deliberately NOT translated.
- **`ChatViewModel.kt`** — the logic: loads a `LlamaModel` (SAF `content://` copied into `filesDir`
  because llama.cpp mmaps a real path), streams via `generateChatFlow` and forwards the `GenerationSettings`
  sampling knobs to `InferenceParameters`; supports **stop** (`generation?.cancel()` — cancellation keeps
  the partial reply, not an error) and **regenerate** (drop the trailing reply, re-run the last user turn,
  shared `startGeneration` path); keeps a capped rolling in-app **log**. **`SessionStore.kt`** — private
  local save/load: the conversation + model path as JSON in `filesDir/session.json` (`org.json`, no dep
  added), readable only by the app. **`LlmServiceApp.kt`** — `Application` that **wipes the transient
  working data** (the copied model `current-model.gguf` + cache) on every **cold start**
  (`onCreate`) — a privacy guarantee independent of the OS calling `onDestroy`; `MainActivity` also
  wipes best-effort on finish. Only the **opt-in saved session** persists. **`MainActivity.kt`** —
  Compose UI with a **two-row app bar** (row 1 = horizontally-scrollable model name + a ❌ unload
  button when a model is loaded, row 2 = all action icons) + SAF picker + flag menu + Save/Load + the ⚙️ Settings sheet
  (sampling knobs + a **Model** section: CPU threads / context length applied via **Reload model**) +
  the 🧾 log strip/viewer + Stop/Copy/Regenerate/Clear chat actions (+ **long-press any bubble to
  copy**) + **prompt shortcut chips** + a **device-readiness card** (free RAM / storage / battery via
  `DeviceInfo`) on the picker + the offline badge; reads optional `MODEL_PATH` / `CHAT_TEMPLATE`
  intent extras as a **test hook** (the shipping UI never sets them).
- **`app/src/androidTest/kotlin/.../ChatFlowInstrumentedTest.kt`** — the real end-to-end test:
  launches the activity with a preloaded model (bypassing the system SAF dialog, which only
  UiAutomator could drive), types a prompt, taps Send, asserts a non-empty streamed reply.
  Self-skips (JUnit `Assume`) when the adb-pushed model is absent.

**Requirements / roadmap docs (keep in sync with the code).** The app has **no unit tests** for its
logic (only the one on-device `ChatFlowInstrumentedTest` above), so
[`android-llmservice/requirements.md`](requirements.md) is the **spec of record** —
it enumerates, with stable `Rn.m` IDs, every feature the app implements today. Any change to app
behavior must update `requirements.md` in the same commit (add/amend/retire a row); the roadmap of
not-yet-built features lives in [`android-llmservice/TODO.md`](TODO.md), and
`README.md`'s feature claims must match `requirements.md`.

**Recommended real model: Gemma 3 4B Instruct** (a `*-it` GGUF; carries its own chat template, so
no override needed). Any instruct GGUF works. The CI on-device test uses the tiny cached draft model
with a forced `chatml` template just to prove tokens generate.

**Key signing fact:** the Maven Central **GPG key cannot sign an APK/AAB** (different
cryptosystem). Android needs a Java keystore (PKCS12/JKS + RSA) **upload key**; Play App Signing
manages the final app-signing key. CI wires it from the optional `ANDROID_UPLOAD_KEYSTORE_BASE64`
+ password/alias secrets.

**CI wiring** — **two split jobs** in `.github/workflows/publish.yml` (both `needs:` the two Android
native jobs; the test job additionally `needs: verify-model-cache`), so the installable artifacts are
**always** produced even when the on-device UI test is flaky/slow:
- **`build-android-llmservice`** — installs the core + `llama-kotlin` to mavenLocal (`llama-kotlin`'s
  core dep is provided-scope, so the desktop JAR is never pulled into the APK), stages the CI-built
  Android natives, publishes the CPU AAR to mavenLocal, builds the **release `.aab`** (signed with the
  upload key when the `ANDROID_UPLOAD_KEYSTORE_BASE64` secret is present, else debug-signed) + the
  installable **`.apk`**, and uploads them (`android-llmservice-aab` / `android-llmservice-apk`).
  **No emulator** — a flaky emulator can never block getting the APK.
- **`test-android-llmservice`** — a **separate, non-gating** check that repeats the setup, then boots
  the KVM x86_64 emulator and runs the Compose UI test via **`.github/run-android-llmservice-test.sh`**
  (adb-pushes the cached `AMD-Llama-135m` draft model, `connectedDebugAndroidTest`). It can go **red on
  its own** without stopping the build/artifacts; keep it **non-required** in branch protection to keep
  it optional (visible-but-non-blocking).

Both emulator jobs (`test-android-llmservice` + `test-android-emulator`) prepend a **free-disk step**
before the emulator (delete every restored model except `DRAFT_MODEL_NAME` + large unused preinstalled
toolchains — the latter via `ggml-org/free-disk-space` with `android`, `large-packages`, `tool-cache`
and `swap-storage` switched **off**, since the emulator needs the SDK, mesa and the JDK) because the AVD userdata partition needs ~7.4 GB and the full ~10 GB GGUF cache restore
otherwise FATALs the emulator ("Not enough space to create userdata partition"). Neither
llmservice job is **yet a publish gate** (not in the `publish-snapshot`/`publish-release` `needs:`
graphs) so a Compose/AGP version-pin hiccup can't block a library release.
`android-llmservice/**/build/` is git-ignored.
