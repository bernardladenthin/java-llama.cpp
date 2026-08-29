<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT
-->
# llama-kotlin

Kotlin coroutines façade for [java-llama.cpp](https://github.com/bernardladenthin/java-llama.cpp):
`Flow`-based token streaming and `suspend` wrappers over the `net.ladenthin:llama` JNI API.
Pure Kotlin/JVM — works on desktop JVMs **and** Android.

```kotlin
dependencies {
    implementation("net.ladenthin:llama-kotlin:5.1.0")
    // ...plus the binding itself — the façade does NOT drag it in transitively
    // (provided scope), so YOU pick the right flavor:
    implementation("net.ladenthin:llama:5.1.0")          // desktop JVM
    // implementation("net.ladenthin:llama-android:5.1.0") // Android (AAR)
}
```

## API

```kotlin
import net.ladenthin.llama.kotlin.*

// Token streaming as a cold Flow — the native task slot is released on
// completion, error, AND cancellation (take(n), Job.cancel, ...):
model.generateChatFlow(params)
    .flowOn(Dispatchers.IO)
    .collect { output -> print(output.text) }

// Suspend wrappers (main-safe; default dispatcher = Dispatchers.IO):
val text: String = model.completeSuspend(params)      // coroutine cancel → native cancel
val chat: ChatResponse = model.chatSuspend(request)
val reply: String = model.chatCompleteTextSuspend(params)
val vector: FloatArray = model.embedSuspend("hello")
```

`completeSuspend` wires **coroutine cancellation to the binding's cooperative
`CancellationToken`**: cancelling the calling coroutine stops the native generation at the next
token boundary and frees the slot — the missing piece a hand-rolled
`withContext(Dispatchers.IO) { model.complete(params) }` does not give you.

## Why `provided` scope on the core?

The desktop `net.ladenthin:llama` JAR bundles native libraries for every desktop platform as Java
resources. If this module depended on it transitively, every Android APK using the façade would
package ~70 MB of dead desktop natives. Declaring the binding yourself keeps the choice explicit:
`llama` on the JVM, the `llama-android` AAR on Android.

## Build

Reactor module — built, versioned and released with the core:

```bash
mvn -pl llama-kotlin -am -DskipTests install   # build with the core
mvn -f llama-kotlin/pom.xml test               # 6 model-free unit tests
```

Requires Kotlin 2.3+ in consuming Kotlin projects (compiled with Kotlin 2.4, metadata readable one
minor back); the bytecode targets Java 8, same as the core.
