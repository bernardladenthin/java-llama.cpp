// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.application")
    // AGP 9.0+ built-in Kotlin support replaces the standalone kotlin-android plugin —
    // see the note in settings.gradle.kts.
    id("org.jetbrains.kotlin.plugin.compose")
}

// Injected by CI (-PjllamaVersion=<reactor version>) so the app always runs against the AAR
// + facade that were just built and published to mavenLocal. Falls back to the last release
// when built by hand / in Android Studio without the property.
val jllamaVersion: String =
    providers.gradleProperty("jllamaVersion").getOrElse("5.0.6")

// -- Release signing (Play upload key) -------------------------------------------------
// The APK/AAB signing key is a Java keystore (PKCS12/JKS + RSA), NOT the Maven Central
// GPG key — different cryptosystem, different tooling. For "as close to a real Play
// release as possible" this reads a dedicated UPLOAD keystore (Play App Signing manages
// the final app-signing key) from env vars (CI secrets) or -P properties (local).
// When no keystore is provided the release build falls back to debug signing, so PR CI,
// forks and local builds still produce an installable APK without any secret.
fun secret(env: String, prop: String): String? =
    System.getenv(env) ?: (findProperty(prop) as String?)

val uploadStorePath: String? = secret("JLLAMA_UPLOAD_STORE_FILE", "jllamaUploadStoreFile")
val hasUploadKeystore: Boolean = !uploadStorePath.isNullOrBlank() && file(uploadStorePath).exists()

android {
    namespace = "net.ladenthin.android.llmservice"
    // 37 (not 35): the AAR-metadata check fails otherwise — activity-compose 1.13.0 /
    // androidx.core 1.18.0 / navigationevent 1.0.0 require compileSdk >= 36, and
    // lifecycle-{viewmodel,runtime}-compose 2.11.0 require compileSdk >= 37. targetSdk is
    // left at 35 on purpose (compileSdk widens the compile-time API surface; targetSdk
    // opts into new runtime behavior and is bumped independently).
    compileSdk = 37

    defaultConfig {
        applicationId = "net.ladenthin.android.llmservice"
        minSdk = 28 // AAR floor (bionic weak-symbol gate for posix_spawn); AGP enforces it.
        targetSdk = 35
        // Monotonic versionCode so Android accepts IN-PLACE UPGRADES: a new APK must advertise a
        // strictly higher code than the installed one, or the install is rejected. CI passes the
        // workflow run number (GITHUB_RUN_NUMBER, strictly increasing per run); `-PappVersionCode`
        // overrides it; a local hand build falls back to 1. (The other half of clean upgrades is a
        // STABLE signing key across builds — see the signing block below and README "Signing".)
        versionCode = (
            providers.gradleProperty("appVersionCode").orNull
                ?: System.getenv("GITHUB_RUN_NUMBER")
        )?.toIntOrNull() ?: 1
        versionName = "1.0"
        // arm64 = real phones; x86_64 = the CI emulator (and x86_64 Android hardware).
        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    signingConfigs {
        if (hasUploadKeystore) {
            create("upload") {
                storeFile = file(uploadStorePath!!)
                storePassword = secret("JLLAMA_UPLOAD_STORE_PASSWORD", "jllamaUploadStorePassword")
                keyAlias = secret("JLLAMA_UPLOAD_KEY_ALIAS", "jllamaUploadKeyAlias")
                keyPassword = secret("JLLAMA_UPLOAD_KEY_PASSWORD", "jllamaUploadKeyPassword")
            }
        }
    }

    buildTypes {
        release {
            // R8 on; the AAR ships consumer ProGuard rules that keep the JNI surface,
            // so no app-side keep rules are needed (proguard-rules.pro is a stub).
            isMinifyEnabled = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
            signingConfig =
                if (hasUploadKeystore) {
                    signingConfigs.getByName("upload")
                } else {
                    logger.lifecycle("No upload keystore configured -> release signed with the debug key.")
                    signingConfigs.getByName("debug")
                }
        }
    }

    buildFeatures {
        compose = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

kotlin {
    // Kotlin 2.x: the old `android { kotlinOptions { jvmTarget = "17" } }` string DSL is now a
    // hard error — configure the JVM target through the compilerOptions DSL instead.
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

dependencies {
    // The binding (classes + libjllama.so for arm64-v8a + x86_64) and the Kotlin coroutines
    // facade (generateChatFlow). The facade's core dependency is provided-scope, so Gradle
    // does NOT pull the fat desktop JAR — the AAR supplies net.ladenthin.llama.* on Android.
    implementation("net.ladenthin:llama-android:$jllamaVersion")
    implementation("net.ladenthin:llama-kotlin:$jllamaVersion")

    // Coroutines (matches the facade's kotlinx-coroutines 1.11.0).
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.11.0")

    // AppCompat: only for its per-app language API (AppCompatDelegate.setApplicationLocales),
    // which powers the in-app flag language picker and persists the choice across restarts.
    implementation("androidx.appcompat:appcompat:1.8.0")

    // Jetpack Compose UI (BOM pins the mutually consistent runtime versions).
    val composeBom = platform("androidx.compose:compose-bom:2026.09.00")
    implementation(composeBom)
    implementation("androidx.activity:activity-compose:1.13.0")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.11.0")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.11.0")
    debugImplementation("androidx.compose.ui:ui-tooling")

    // On-emulator instrumentation (build-android-llmservice CI job).
    androidTestImplementation(composeBom)
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")
    androidTestImplementation("androidx.test.ext:junit:1.3.0")
    androidTestImplementation("androidx.test:core:1.7.0")
    androidTestImplementation("androidx.test:runner:1.7.0")
    debugImplementation("androidx.compose.ui:ui-test-manifest")
}
