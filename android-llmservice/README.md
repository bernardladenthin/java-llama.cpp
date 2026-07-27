<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT
-->
# LLM Service — Android example app

A minimal, **KISS, fully-offline local-AI** Android chat app built on
[`net.ladenthin:llama-android`](../llama-android) (the AAR) and
[`net.ladenthin:llama-kotlin`](../llama-kotlin) (the coroutines façade). It does exactly
what it says:

1. **Pick a GGUF model** from the device's file system (Storage Access Framework — no
   storage permission, Google-Play compliant).
2. **Chat** with it, fully **on-device**, tokens streaming into a Jetpack Compose UI.
3. Optionally add a **vision projector (mmproj)** and attach an **image** (📎) to a
   message — for multimodal GGUFs such as Gemma 3 4B (vision) or SmolVLM2.
4. **Switch language** via a flag picker, and **Save/Load** the conversation locally.

- **App label:** `LLM Service`
- **applicationId / package:** `net.ladenthin.android.llmservice`
- **No `INTERNET` permission, no storage permission** — nothing leaves the device.

> Minimum Android API **28** (Android 9 Pie) — the AAR's floor. Ships native code for
> `arm64-v8a` (phones) and `x86_64` (emulators / x86 Android).

## Recommended model: Gemma 3 4B

Use an **instruct/chat** GGUF for coherent replies. A good default is **Gemma 3 4B
Instruct** (a `*-it` GGUF, ~2.5–3.5 GB depending on quantization) — it carries its own
chat template, so the app's chat path works without any override. Any instruct GGUF
(Qwen, Llama, Phi, …) works too. Base/completion-only models will still run but won't
behave like a chat assistant. (The CI on-device test uses a tiny draft model with a forced
`chatml` template just to prove tokens generate — see below.)

## Internationalization (13 languages + flag picker)

Every user-facing string is a resource; translations ship for **English, German, Spanish,
French, Italian, Portuguese, Russian, Turkish, Arabic (RTL), Hindi, Chinese (Simplified),
Japanese, Korean** (`app/src/main/res/values-*/strings.xml`). A flag-emoji dropdown in the
top bar switches the language **in-app** using AndroidX per-app locales
(`AppCompatDelegate.setApplicationLocales`), which persists the choice across restarts and
integrates with the Android 13+ system per-app language setting (`res/xml/locales_config.xml`).
The localized system prompt also nudges the model to answer in the chosen language.
(Flags label languages for friendliness — a flag is a country, not a language.)

## Private save / load

The **Save** (💾) and **Load** (📂) buttons persist the conversation — and the model's
on-device path — to a single JSON file in the app's **private internal storage**
(`filesDir/session.json`), readable only by this app and never uploaded. It's opt-in
(nothing autosaves) and fully local, keeping the "nothing leaves the device" promise.
Loading restores the messages and re-opens the model if its file is still present.

## Ephemeral by default (privacy)

Nothing lingers between sessions. The copied model (`current-model.gguf`, which can be several
GB) and the app cache are **wiped on every cold start** (`LlmServiceApp.onCreate`) — a guarantee
that holds no matter how the app was last closed (swipe-away, low-memory kill, crash) — and are
also cleared best-effort when the app is finished (`MainActivity.onDestroy`). So after you fully
close the app you re-pick the model on next launch, and no large data sits around. The **only**
thing that persists is the session you **explicitly Save** (see below) — everything else is
transient. (A language switch recreates the activity but does *not* wipe the in-use model.)

## Why a `content://` URI is copied to a real path

llama.cpp memory-maps the model from a **real filesystem path**, but the file picker
returns a `content://` URI. So `ChatViewModel` copies the picked file into the app's
private `filesDir` and loads it by path. This is the Play-safe route — no
`MANAGE_EXTERNAL_STORAGE`, no `READ_EXTERNAL_STORAGE`.

## Build it — no Android Studio required

Android Studio is only an IDE; the build is plain Gradle + AGP. You need a JDK 17 and the
Android SDK command-line tools (`sdkmanager`) with a platform + build-tools; point Gradle
at the SDK via `local.properties` (`sdk.dir=/path/to/Android/sdk`) or the
`ANDROID_HOME`/`ANDROID_SDK_ROOT` env var. Then, from the repo root:

```bash
# The app resolves the AAR + Kotlin façade from mavenLocal, so publish them first
# (or depend on a released version from Maven Central and skip these two lines).
mvn -q -pl llama,llama-kotlin -am -DskipTests -Dgpg.skip=true install
#   ...plus stage the Android natives + `gradle -p llama-android publishToMavenLocal`
#   (CI does this automatically — see .github/workflows/publish.yml build-android-llmservice).

VERSION=$(mvn -q -DforceStdout help:evaluate -Dexpression=project.version -pl llama | tail -n1)

# Debug APK (installable on a device/emulator):
gradle -p android-llmservice assembleDebug -PjllamaVersion="$VERSION"

# Release App Bundle for Google Play (.aab):
gradle -p android-llmservice bundleRelease -PjllamaVersion="$VERSION"
```

`-PjllamaVersion` selects which `llama-android`/`llama-kotlin` version to consume; omit it
to use the default pinned in `app/build.gradle.kts`. Opening the `android-llmservice/`
folder in Android Studio works too — it's a normal Gradle project.

## Signing & Google Play

**The Maven Central GPG key cannot sign an APK/AAB** — Android signing uses a Java keystore
(PKCS12/JKS + RSA), a different cryptosystem. For Play, the recommended model is **Play App
Signing**: you create a one-time **upload keystore**, sign the AAB with it, and Google
manages the final app-signing key. Create the upload key once:

```bash
keytool -genkeypair -v -keystore upload-keystore.jks -alias upload \
        -keyalg RSA -keysize 4096 -validity 10000
```

`app/build.gradle.kts` reads the keystore + passwords from env vars (or `-P` properties):

| Purpose        | Env var                        | Gradle property             |
|----------------|--------------------------------|-----------------------------|
| Keystore path  | `JLLAMA_UPLOAD_STORE_FILE`     | `jllamaUploadStoreFile`     |
| Store password | `JLLAMA_UPLOAD_STORE_PASSWORD` | `jllamaUploadStorePassword` |
| Key alias      | `JLLAMA_UPLOAD_KEY_ALIAS`      | `jllamaUploadKeyAlias`      |
| Key password   | `JLLAMA_UPLOAD_KEY_PASSWORD`   | `jllamaUploadKeyPassword`   |

When no keystore is supplied, the release build falls back to **debug signing** so local
builds, forks, and PR CI still produce an installable artifact without any secret.

```bash
JLLAMA_UPLOAD_STORE_FILE=$PWD/upload-keystore.jks \
JLLAMA_UPLOAD_STORE_PASSWORD=… JLLAMA_UPLOAD_KEY_ALIAS=upload JLLAMA_UPLOAD_KEY_PASSWORD=… \
  gradle -p android-llmservice bundleRelease -PjllamaVersion="$VERSION"
# -> app/build/outputs/bundle/release/app-release.aab  (upload to the Play Console)
```

The first upload of a new package to the Play Console must be done manually (to register the
app + enroll in Play App Signing). After that, automated Play uploads are possible with the
Gradle Play Publisher plugin or fastlane `supply` (needs a Play service-account JSON) — left
out here to keep the demo KISS.

### In-place upgrades (why "problem parsing the package" / "app not installed" on update)

Android only installs an update **over** an existing app when **both** hold:

1. **Same signing key.** A different signer is rejected (often surfaced as the generic
   *"There was a problem parsing the package"* / *"App not installed"*) — you must uninstall
   the old app first. This is exactly what happens with the **debug-signing fallback**: a
   GitHub-hosted runner has no persistent debug keystore, so **each CI build auto-generates a
   new random debug key** → consecutive sideloaded APKs have **different** signatures.
   **Fix:** set the four `ANDROID_UPLOAD_KEYSTORE_BASE64` / `…_STORE_PASSWORD` / `…_KEY_ALIAS`
   / `…_KEY_PASSWORD` CI secrets (above) so every build is signed with the **same** upload key.
   Keep that keystore private and backed up — never commit it (the repo is public; a leaked
   signing key lets anyone ship a fake update over your users' install).
2. **Higher `versionCode`.** The new APK must advertise a strictly greater `versionCode`.
   `app/build.gradle.kts` derives it from `GITHUB_RUN_NUMBER` in CI (strictly increasing per
   run; `-PappVersionCode=<n>` overrides; local hand builds use `1`), so successive CI APKs
   upgrade cleanly once the signing key is stable.

Until the upload-key secrets are configured, sideloaded CI builds are debug-signed with an
ephemeral key and will keep requiring an **uninstall before reinstall**. (Also make sure you
install the `.apk` artifact, not the `.aab` — an `.aab` is not directly installable and also
fails to parse.)

## CI: build, sign, and a **real** on-device test

The `build-android-llmservice` job in `.github/workflows/publish.yml`:

1. Builds the core jar + installs `llama-kotlin` to mavenLocal, stages the CI-built Android
   natives, and publishes the `llama-android` AAR to mavenLocal.
2. Builds a **release `.aab`** — signed with the upload key when the CI secrets are set
   (`ANDROID_UPLOAD_KEYSTORE_BASE64` = base64 of the keystore, plus the store/alias/key
   password secrets), otherwise debug-signed — and uploads it as the `android-llmservice-aab`
   artifact. It also uploads a **directly-installable release `.apk`** as the
   `android-llmservice-apk` artifact: download it from the run's **Artifacts** section
   (independent of any release) and sideload it with `adb install app-release.apk`. (The `.aab`
   is only for the Play Console — it is not directly installable; the `.apk` is the one to grab
   to try the app on a phone.)
3. Runs the app's **instrumented UI test on a KVM-accelerated x86_64 emulator** (API 30): it
   adb-pushes the tiny cached draft model, then `connectedDebugAndroidTest` launches the app,
   **types a prompt, taps Send, and asserts a non-empty assistant reply** — exercising the
   Compose UI, the `ChatViewModel`, the `generateChatFlow` façade, and real native inference
   together.

### Why an instrumented (emulator) test, not Espresso-in-a-vacuum

The valuable test here is the **on-device** one — Espresso / Compose UI tests only mean
something when run on a real device or emulator (`connectedAndroidTest`), which is exactly
what the CI job does. This app uses a **Compose UI test** (`createEmptyComposeRule` +
`ActivityScenario`) rather than classic Espresso, but they're interchangeable for driving the
UI. The one thing deliberately **not** UI-tested is the system file-picker dialog: it runs in
a separate process (only `UiAutomator` could drive it, and it's flaky), so the test injects the
model path via an `Intent` extra and drives everything after the picker. See
`app/src/androidTest/.../ChatFlowInstrumentedTest.kt`.

Run the same test locally against a booted emulator/device:

```bash
adb push <tiny-model>.gguf /data/local/tmp/jllama-test-model.gguf
adb shell chmod 644 /data/local/tmp/jllama-test-model.gguf
gradle -p android-llmservice connectedDebugAndroidTest -PjllamaVersion="$VERSION"
# The test self-skips (JUnit Assume) if the model file is absent.
```

## Layout

```
android-llmservice/
├── settings.gradle.kts          # AGP 9.2.1 (built-in Kotlin) + Compose plugin 2.4.10, mavenLocal first
├── gradle.properties
├── requirements.md              # spec of record: every feature the app implements today (no unit tests)
├── TODO.md                      # roadmap: not-yet-built features
└── app/
    ├── build.gradle.kts         # minSdk 28, Compose, release signingConfig (upload key/debug fallback)
    ├── proguard-rules.pro       # stub — the AAR ships the JNI keep rules
    └── src/
        ├── main/AndroidManifest.xml
        ├── main/res/values/{strings,themes}.xml + values-*/strings.xml (13 languages)
        ├── main/res/xml/locales_config.xml
        ├── main/kotlin/net/ladenthin/android/llmservice/
        │   ├── MainActivity.kt      # Compose UI (two-row app bar) + SAF picker + flag menu + save/load + settings + log + chat actions + prompt chips + device card
        │   ├── LlmServiceApp.kt     # Application: wipes transient working data (model copy + cache) on cold start (privacy)
        │   ├── ChatViewModel.kt     # model load + streaming chat (stop/regenerate) + sampling & model settings + in-app log + session (the logic)
        │   ├── DeviceInfo.kt        # device-readiness snapshot (free RAM / storage / battery) for the picker card
        │   ├── Languages.kt         # the flag/language list
        │   └── SessionStore.kt      # private local JSON persistence (filesDir)
        └── androidTest/kotlin/net/ladenthin/android/llmservice/
            └── ChatFlowInstrumentedTest.kt   # end-to-end on-device UI test
```
