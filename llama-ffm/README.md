<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT
-->
# llama-ffm

[java-llama.cpp](https://github.com/bernardladenthin/java-llama.cpp) without JNI: the C API of
`libjllama` ([`jllama_api.h`](../docs/c-api.md), shipped in every natives directory) called through
the Java 22+ **Foreign Function and Memory API**. The same natives jars as the core, found and loaded
by the core's `LlamaLoader`; this module only resolves the `jllama_*` symbols the loaded library
exports. One implementation behind both call paths: what the core's JNI entry points do, these
classes do through the C functions the JNI shells call.

```xml
<dependency>
    <groupId>net.ladenthin</groupId>
    <artifactId>llama-ffm</artifactId>
    <version>6.0.0</version>
</dependency>
<!-- plus the natives, exactly as for the core: llama-platform (every desktop CPU) ... -->
<dependency>
    <groupId>net.ladenthin</groupId>
    <artifactId>llama-platform</artifactId>
    <version>6.0.0</version>
    <type>pom</type>
</dependency>
<!-- ... and a GPU module jar where wanted, e.g. net.ladenthin:llama:6.0.0:cuda13-linux-x86-64 -->
```

Java 22 or newer (the FFM API is final from 22 on); run the JVM with
`--enable-native-access=ALL-UNNAMED` (or name the module) to silence the restricted-method warning.
Not for Android, which has no FFM API.

## API

Requests and responses are JSON text, the bodies llama.cpp's own HTTP server reads and writes;
`ModelParameters` renders the argv and `InferenceParameters.toJson()` a request body, as for the core.
A request llama.cpp rejects throws the core's `InvalidRequestException`, any other failure its
`LlamaException`.

```java
import net.ladenthin.llama.ffm.*;
import net.ladenthin.llama.parameters.ModelParameters;

try (FfmLlamaModel model = FfmLlamaModel.load(new ModelParameters().setModel("model.gguf").setGpuLayers(99))) {
    String answer = model.chatCompletions(
            "{\"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}], \"max_tokens\": 64}");

    try (FfmTaskStream stream = model.chatCompletionsStream("{\"messages\": [...], \"stream\": true}")) {
        while (stream.hasNext()) {
            String envelope = stream.next(); // {"data": <chunk>, "stop": <bool>}
        }
    }

    int[] tokens = model.tokenize("Hello", false);
    String text = model.detokenize(tokens);
    String metrics = model.metrics();

    // The embedded llama-server over the loaded model, WebUI included:
    try (FfmNativeServer server = FfmNativeServer.startAttached(model, "--port", "9931")) {
        // ... serve ...
    }
}

Jllama.setLogCallback((level, line) -> System.err.print(level + " " + line));
String grammar = Jllama.jsonSchemaToGrammar("{\"type\": \"object\"}");
```

| Class | C functions |
|---|---|
| `FfmLlamaModel` | `jllama_model_*`, chat / raw / OpenAI completions and their streams, `apply_template`, `embed`, `embeddings`, `rerank`, `tokenize`, `detokenize`, `metrics`, `slot_action`, LoRA, `configure`, `systemone` |
| `FfmTaskStream` | `jllama_*_stream_receive`, `jllama_task_release` |
| `FfmNativeServer` | `jllama_server_*` (classic, attached, the router's worker command) |
| `FfmRpcServer` | `jllama_rpc_server_*` |
| `FfmTextToSpeech` | `jllama_tts_*` |
| `Jllama` | `jllama_api_version`, `jllama_build_info`, `jllama_set_log_callback`, `jllama_json_schema_to_grammar`, `jllama_quantize`, `jllama_finetune` |

## Build

Reactor module — built, versioned and released with the core. The reactor uses JDK 25 from
`.java-version`; this module still targets Java 22 and the core still targets Java 8.
The `toolchain-jdk22` profile remains a fallback for developers running Maven on an older JDK:
it selects a registered JDK 22+ from `~/.m2/toolchains.xml`. CI needs only the shared JDK.

```bash
mvn -pl llama -am -DskipTests install          # the core first
mvn -f llama-ffm/pom.xml verify                # the model-free tests + the model-backed ones
# the model-backed test takes the draft model of .github/models.csv, or:
mvn -f llama-ffm/pom.xml verify -Dnet.ladenthin.llama.ffm.model=/path/to/model.gguf
```

`FfmNativeServerTest` and `FfmTextToSpeechTest` run over fake `Backend` implementations and need
no native library. `JllamaNativeTest` and the model-backed `FfmLlamaModelIntegrationTest` skip
themselves without a native build (`llama/src/main/natives`, or
`-Dnet.ladenthin.llama.lib.path=<natives directory>`); `FfmRpcServerTest` serves the CPU over
loopback and needs one as well.
