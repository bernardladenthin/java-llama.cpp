// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

// CI fixture (NOT a shipped project): a minimal AGP app that consumes the
// net.ladenthin:llama-android AAR from mavenLocal and runs a full R8 release
// build — proving Android Studio consumption end-to-end (AAR format, manifest
// minSdk merge, jni packaging, consumer proguard rules) without an emulator.
// Driven by the package-android-aar job in .github/workflows/publish.yml.
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
    plugins {
        // AGP 9.3.x requires Gradle >= 9.5.0 — see the gradle-version pins on the CI jobs
        // that build this fixture in publish.yml (package-android-aar, test-android-emulator).
        // Stay at >= 9.2.1: 9.2.1 (not 9.2.0) first fixed a real R8 regression
        // (java.lang.ClassNotFoundException on com.android.tools.r8.RecordTag after
        // upgrading Gradle to 9.x with AGP 9.2.0) that hits this fixture directly since
        // buildTypes.release sets isMinifyEnabled = true; 9.3.0 carries that fix forward.
        id("com.android.application") version "9.3.0"
    }
}

dependencyResolutionManagement {
    repositories {
        mavenLocal() // the freshly built llama-android AAR is published here first
        google()
        mavenCentral()
    }
}

rootProject.name = "llama-android-consumer-test"
include(":app")
