// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

// Builds the Android AAR artifacts for java-llama.cpp WITHOUT the Android Gradle
// Plugin and without an Android SDK:
//
//   net.ladenthin:llama-android          — CPU natives (arm64-v8a)
//   net.ladenthin:llama-android-opencl   — OpenCL/Adreno natives (arm64-v8a)
//
// An AAR is a documented zip (AndroidManifest.xml + classes.jar + jni/<abi>/ +
// proguard.txt + R.txt). AGP is only required to *consume* it — which the CI
// consumer smoke test does on a runner with the Android SDK. Building it here
// with plain Gradle means:
//   1. classes.jar carries the BYTE-IDENTICAL Maven-built core classes (no
//      recompilation, no Lombok/AGP coupling, no drift from the tested jar);
//      only module-info.class is stripped. The core jar carries no natives (they
//      ship as separate natives jars); here the .so ships under jni/ instead —
//      LlamaLoader calls System.loadLibrary("jllama") first on Android.
//   2. The published POM says <packaging>aar</packaging>, which plain Maven
//      cannot produce — this is exactly how AGP-built libraries on Central
//      declare themselves, so `implementation("net.ladenthin:llama-android:V")`
//      resolves in any Android project without an @aar suffix.
//
// Version lockstep: the version and all mirrored dependency versions are parsed
// from the Maven poms at configure time. `mvn versions:set` (the documented bump
// procedure) is the only version edit point; this build follows automatically.
//
// Inputs expected before running the aar tasks (fail-loud checks below):
//   ../llama/target/llama-<version>.jar       mvn -pl llama -am -DskipTests package
//   natives/cpu/arm64-v8a/libjllama.so        CI artifact natives-cpu-android-aarch64
//   natives/cpu/x86_64/libjllama.so           CI artifact natives-cpu-android-x86-64
//   natives/opencl/arm64-v8a/libjllama.so     CI artifact natives-opencl-android-aarch64

import org.w3c.dom.Document
import org.w3c.dom.Element
import org.w3c.dom.Node
import javax.xml.parsers.DocumentBuilderFactory

plugins {
    base
    `maven-publish`
    signing
}

// ---------------------------------------------------------------------------
// Single-source-of-truth version/metadata parsing from the Maven reactor poms.
// ---------------------------------------------------------------------------

fun parsePom(path: String): Document {
    val factory = DocumentBuilderFactory.newInstance()
    factory.isNamespaceAware = false
    return factory.newDocumentBuilder().parse(file(path))
}

fun directChildText(element: Element, tag: String): String? {
    val children = element.childNodes
    for (i in 0 until children.length) {
        val node = children.item(i)
        if (node.nodeType == Node.ELEMENT_NODE && node.nodeName == tag) {
            return node.textContent.trim()
        }
    }
    return null
}

fun directChildElement(element: Element, tag: String): Element? {
    val children = element.childNodes
    for (i in 0 until children.length) {
        val node = children.item(i)
        if (node.nodeType == Node.ELEMENT_NODE && node.nodeName == tag) {
            return node as Element
        }
    }
    return null
}

val rootPom = parsePom("../pom.xml")
val corePom = parsePom("../llama/pom.xml")

val reactorVersion = directChildText(rootPom.documentElement, "version")
    ?: error("No <version> found in ../pom.xml — the root reactor pom must declare the version")

fun coreProperty(name: String): String {
    val properties = directChildElement(corePom.documentElement, "properties")
        ?: error("No <properties> in ../llama/pom.xml")
    return directChildText(properties, name)
        ?: error("Property <$name> not found in ../llama/pom.xml <properties> — keep the AAR pom dependencies in lockstep with the core")
}

group = "net.ladenthin"
version = reactorVersion

val coreJarFile = file("../llama/target/llama-$reactorVersion.jar")

// ---------------------------------------------------------------------------
// AAR content tasks.
// ---------------------------------------------------------------------------

// The AAR classes.jar: the Maven-built core jar minus (a) every bundled desktop
// native resource tree (they would otherwise be packaged into consumer APKs —
// ~70 MB of dead weight; the Android .so ships under jni/ instead) and
// (b) module-info.class (D8 does not accept JPMS descriptors in classes.jar).
val coreClassesJar = tasks.register<Jar>("coreClassesJar") {
    description = "Repackages the Maven-built core classes as the AAR classes.jar payload."
    archiveBaseName.set("classes-payload")
    destinationDirectory.set(layout.buildDirectory.dir("intermediates"))
    isPreserveFileTimestamps = false
    isReproducibleFileOrder = true
    doFirst {
        require(coreJarFile.isFile) {
            "Core jar not found: $coreJarFile — build it first: mvn -pl llama -am -DskipTests package (from the repo root)"
        }
    }
    from(zipTree(coreJarFile)) {
        exclude("module-info.class")
        exclude("META-INF/maven/**")
    }
}

// AGP-built AARs always carry an R.txt (empty for resource-less libraries).
val generateRTxt = tasks.register("generateRTxt") {
    val rTxt = layout.buildDirectory.file("intermediates/R.txt")
    outputs.file(rTxt)
    doLast { rTxt.get().asFile.writeText("") }
}

val sourcesJar = tasks.register<Jar>("sourcesJar") {
    description = "Core Java sources (Central requires a -sources jar per artifact)."
    archiveClassifier.set("sources")
    isPreserveFileTimestamps = false
    isReproducibleFileOrder = true
    from("../llama/src/main/java") {
        exclude("module-info.java")
    }
}

val javadocJar = tasks.register<Jar>("javadocJar") {
    description = "Javadoc placeholder jar (Central requires a -javadoc jar per artifact; " +
            "the full javadoc ships with net.ladenthin:llama, whose classes this AAR repackages)."
    archiveClassifier.set("javadoc")
    val readme = layout.buildDirectory.file("intermediates/javadoc-readme/README.txt")
    doFirst {
        val file = readme.get().asFile
        file.parentFile.mkdirs()
        file.writeText(
            "This artifact repackages the net.ladenthin:llama classes for Android.\n" +
                    "The full javadoc is published with net.ladenthin:llama:$reactorVersion (classifier 'javadoc')\n" +
                    "and online via javadoc.io.\n"
        )
    }
    from(readme.map { it.asFile.parentFile })
}

fun registerAarTask(taskName: String, artifactBase: String, nativesSubdir: String, requiredAbis: List<String>) =
    tasks.register<Zip>(taskName) {
        description = "Assembles $artifactBase-$reactorVersion.aar from the core classes and natives/$nativesSubdir."
        archiveBaseName.set(artifactBase)
        archiveVersion.set(reactorVersion)
        archiveExtension.set("aar")
        destinationDirectory.set(layout.buildDirectory.dir("aar"))
        isPreserveFileTimestamps = false
        isReproducibleFileOrder = true
        val nativesDir = file("natives/$nativesSubdir")
        doFirst {
            // Fail-loud per ABI: a missing staging copy must never silently produce an
            // AAR that lacks an advertised ABI (emulator/x86_64 consumers would crash
            // at load time instead).
            for (abi in requiredAbis) {
                val so = File(nativesDir, "$abi/libjllama.so")
                require(so.isFile) {
                    "Missing Android native library: $so — stage the CI-built libjllama.so there " +
                            "(artifacts 'natives-cpu-android-aarch64' / 'natives-cpu-android-x86-64' " +
                            "for cpu, 'natives-opencl-android-aarch64' for opencl; the artifact tree is " +
                            "<OS>/<arch>/<backend>/libjllama.so, e.g. Linux-Android/aarch64/cpu/libjllama.so)"
                }
            }
        }
        from("src/main/AndroidManifest.xml")
        from(coreClassesJar) { rename { "classes.jar" } }
        from("consumer-proguard.txt") { rename { "proguard.txt" } }
        from(generateRTxt)
        from(nativesDir) { into("jni") }
    }

// CPU AAR is multi-ABI: arm64-v8a for devices, x86_64 for emulators / x86_64 Android
// hardware (Chromebooks etc.). App bundles split per ABI, so phones download only arm64.
// The OpenCL flavor stays arm64-only (Adreno = Qualcomm ARM hardware).
val aarCpu = registerAarTask("aarCpu", "llama-android", "cpu", listOf("arm64-v8a", "x86_64"))
val aarOpencl = registerAarTask("aarOpencl", "llama-android-opencl", "opencl", listOf("arm64-v8a"))

// ---------------------------------------------------------------------------
// Publishing: POM <packaging>aar</packaging> + mirrored core dependencies.
// ---------------------------------------------------------------------------

// Suppress Gradle Module Metadata: consumers must resolve via the POM
// (<packaging>aar</packaging> → .aar artifact), the exact mechanism every
// pre-GMM Android library on Central uses. Ad-hoc-artifact GMM would lack the
// variant attributes AGP expects and could confuse resolution.
tasks.withType<GenerateModuleMetadata>().configureEach { enabled = false }

fun org.gradle.api.publish.maven.MavenPom.commonMetadata(artifactDisplayName: String, backendNote: String) {
    name.set(artifactDisplayName)
    description.set(
        "Android AAR for java-llama.cpp: the net.ladenthin:llama Java API with the $backendNote " +
                "arm64-v8a native library packaged under jni/, consumer R8/ProGuard rules, and minSdk 28."
    )
    url.set("https://github.com/bernardladenthin/java-llama.cpp")
    licenses {
        license {
            name.set("MIT License")
            url.set("https://opensource.org/licenses/MIT")
        }
    }
    developers {
        developer {
            id.set("bernardladenthin")
            name.set("Bernard Ladenthin")
            email.set("bernard.ladenthin@gmail.com")
        }
    }
    scm {
        connection.set("scm:git:git://github.com/bernardladenthin/java-llama.cpp.git")
        developerConnection.set("scm:git:ssh://github.com:bernardladenthin/java-llama.cpp.git")
        url.set("https://github.com/bernardladenthin/java-llama.cpp")
    }
    // Mirror the core's compile-scope dependencies (the AAR repackages those
    // classes, so their imports must resolve on the consumer classpath).
    // logback-classic is deliberately NOT mirrored: it is the core's JVM-only
    // runtime SLF4J binding and does not run on Android — Android consumers
    // pick their own binding (e.g. slf4j-android) or run with the no-op one.
    withXml {
        val dependencies = asNode().appendNode("dependencies")
        fun dependency(groupId: String, artifactId: String, versionProperty: String) {
            val node = dependencies.appendNode("dependency")
            node.appendNode("groupId", groupId)
            node.appendNode("artifactId", artifactId)
            node.appendNode("version", coreProperty(versionProperty))
            node.appendNode("scope", "compile")
        }
        dependency("com.fasterxml.jackson.core", "jackson-databind", "jackson.version")
        dependency("org.slf4j", "slf4j-api", "slf4j.version")
        dependency("org.jspecify", "jspecify", "jspecify.version")
        dependency("org.checkerframework", "checker-qual", "checker.version")
    }
}

publishing {
    publications {
        create<MavenPublication>("llamaAndroid") {
            artifactId = "llama-android"
            artifact(aarCpu)
            artifact(sourcesJar)
            artifact(javadocJar)
            pom.packaging = "aar"
            pom.commonMetadata("llama-android", "CPU")
        }
        create<MavenPublication>("llamaAndroidOpencl") {
            artifactId = "llama-android-opencl"
            artifact(aarOpencl)
            artifact(sourcesJar)
            artifact(javadocJar)
            pom.packaging = "aar"
            pom.commonMetadata("llama-android-opencl", "OpenCL/Adreno")
        }
    }
    repositories {
        // Local staging repo in Maven layout — CI zips this into a Central
        // Portal bundle for releases, and it doubles as the inspection target
        // for the structural AAR checks.
        maven {
            name = "staging"
            url = uri(layout.buildDirectory.dir("staging-repo"))
        }
        // Central Portal snapshot repository — plain Maven layout, token auth.
        // Only wired when CI provides the credentials.
        val centralUsername = System.getenv("CENTRAL_USERNAME")
        val centralPassword = System.getenv("CENTRAL_PASSWORD")
        if (centralUsername != null && centralPassword != null) {
            maven {
                name = "centralSnapshots"
                url = uri("https://central.sonatype.com/repository/maven-snapshots/")
                credentials {
                    username = centralUsername
                    password = centralPassword
                }
            }
        }
    }
}

// Sign only when CI provides the key (same GPG key the Maven release uses).
val signingKey = System.getenv("MAVEN_GPG_PRIVATE_KEY")
val signingPassphrase = System.getenv("MAVEN_GPG_PASSPHRASE")
// The signing (sub)key id (e.g. 07D2D767). When set, Gradle selects that key
// instead of the primary. This key's signing capability lives on a 4096-bit
// subkey; gpg (maven-gpg-plugin) auto-selects it, but the 2-arg
// useInMemoryPgpKeys picks the primary — whose secret this BouncyCastle can't
// unlock, failing with a null PGPPrivateKey. Selecting the subkey by id fixes it.
val signingKeyId = System.getenv("MAVEN_GPG_KEY_ID")
if (signingKey != null) {
    signing {
        if (!signingKeyId.isNullOrBlank()) {
            useInMemoryPgpKeys(signingKeyId, signingKey, signingPassphrase ?: "")
        } else {
            useInMemoryPgpKeys(signingKey, signingPassphrase ?: "")
        }
        sign(publishing.publications)
    }
}

// The two publications (llamaAndroid / llamaAndroidOpencl) share the SAME sourcesJar/javadocJar,
// so signing produces those shared *.asc files: e.g. the OpenCL publish task reads the CPU
// publication's Sign output without an inferred dependency, which Gradle 8's task-output
// validation rejects as an "implicit dependency" (fails: ":publishLlamaAndroidOpencl…" uses the
// output of ":signLlamaAndroidPublication"). Make every publish task depend on every signing task
// so the ordering is explicit. No-op when signing is off (tasks.withType<Sign>() is then empty).
tasks.withType<org.gradle.api.publish.maven.tasks.AbstractPublishToMaven>().configureEach {
    dependsOn(tasks.withType<org.gradle.plugins.signing.Sign>())
}
