// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

// Standalone plain-Gradle example app (NOT a Maven reactor module, like llama-android
// and .github/android-consumer-test). "LLM Service": a KISS, fully-offline local-AI chat
// demo that consumes the shipped net.ladenthin:llama-android AAR + net.ladenthin:llama-kotlin
// coroutines facade. Built + emulator-tested by the build-android-llmservice job in
// .github/workflows/publish.yml; also opens directly in Android Studio.
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
    // Versions pinned here (KISS: no version catalog). AGP matches the consumer-test
    // fixture. AGP 9.3.x requires Gradle >= 9.5.0 — see the gradle-version pins on the CI
    // jobs that build this project in publish.yml. Stay at >= 9.2.1: 9.2.1 (not 9.2.0)
    // first fixed a real R8 regression (java.lang.ClassNotFoundException on
    // com.android.tools.r8.RecordTag after upgrading Gradle to 9.x with AGP 9.2.0) that
    // hits this project directly since buildTypes.release sets isMinifyEnabled = true;
    // 9.3.0 carries that fix forward.
    // AGP 9.0+ has built-in Kotlin support (runtime dependency on Kotlin Gradle plugin
    // 2.2.10+), so the standalone org.jetbrains.kotlin.android plugin is no longer
    // applied/needed — see https://developer.android.com/build/migrate-to-built-in-kotlin.
    // The Compose compiler plugin (2.4.10) still applies separately and exceeds AGP's
    // 2.2.10 floor.
    plugins {
        id("com.android.application") version "9.3.0"
        id("org.jetbrains.kotlin.plugin.compose") version "2.4.10"
    }
}

dependencyResolutionManagement {
    repositories {
        // In CI the freshly built llama-android AAR + llama-kotlin jar are published to
        // mavenLocal first; for real users these resolve from Maven Central.
        mavenLocal()
        google()
        mavenCentral()
    }
}

rootProject.name = "android-llmservice"
include(":app")
