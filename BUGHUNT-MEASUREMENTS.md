<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT
-->

# Messauftrag zum Bug-Hunt 2026-10-16 … 19 — java-llama.cpp

Dieser Auftrag sammelt jede Messung und jeden modell- oder CI-gestützten Lauf, den der Bug-Hunt offen
gelassen hat. Der Hintergrund steht in `BUGHUNT-FOUND.md` (Fund-IDs, offene Verdachte) und in `TODO.md`
(„Open points from the bug hunt“).

**Regeln für den ausführenden Agenten**

- **Nur messen und berichten.** Keine Produktionsänderung, keine Fixes. Wegwerf-Proben (Testklassen,
  kleine `main`-Programme) nach dem Lauf wieder entfernen; committet wird nur diese Datei mit den
  Ergebnissen. Wo ein Punkt ausdrücklich einen Rot-Test verlangt (J7, J8), den Test schreiben, laufen
  lassen, das Ergebnis eintragen und den Test **nicht** committen; der Quelltext des Tests gehört dann
  ins Ergebnis.
- Arbeitsstand: Branch `ccr-90fc5f6a-vs4s0f`. Pin: llama.cpp b11639.
- Dateien aus `.github/shared-files.sha256` nicht verändern.
- Jeder Punkt wird beantwortet, auch wenn er nicht ausführbar war (mit Grund). Zu jedem Ergebnis die
  Umgebung (CPU, Kerne, RAM, GPU, OS, JDK) und die wörtlichen Ausgaben, die das Ergebnis tragen.
- Ergebnis unter **Ergebnis:**, darunter **Folgerung:** (was es für die genannte Entscheidung heißt).

**Gemeinsame Vorbereitung**

```bash
# JDK: Temurin 21 (nicht Oracle 21, siehe CLAUDE.md); für J7/J8 zusätzlich ein JDK 22+ in ~/.m2/toolchains.xml
mvn -B -q -f llama/pom.xml compile
cd llama && cmake -B build && cmake --build build --config Release -j"$(nproc)" && cd ..
# Modelle aus .github/models.csv nach models/ (die Tests finden sie dort):
mkdir -p models
while IFS=, read -r f url; do case "$f" in \#*|"") continue;; esac
  [ -f "models/$f" ] || curl -L --fail -o "models/$f" "$url"; done < .github/models.csv
# Auf einem Rechner ohne GPU alle Läufe mit -Dnet.ladenthin.llama.test.ngl=0
```

Für die kleinen Messprogramme unten genügt eine Wegwerf-Testklasse unter
`llama/src/test/java/net/ladenthin/llama/` (dann gilt der Test-Klassenpfad mit `src/main/natives/`), gestartet
mit `mvn -B -q -f llama/pom.xml test -Dtest=<Klasse>`. Pfade über `TestConstants` (`DRAFT_MODEL_PATH`,
`REASONING_MODEL_PATH`, `DEFAULT_TOOL_MODEL_PATH`).

---

## Messumgebung (Ergebnisse vom 2026-10-19)

**Umgebung aller Läufe:** Vultr `voc-c-16c-32gb-300s-amd` (Frankfurt, dedizierte Kerne), AMD EPYC-Milan, 16 vCPU,
31 GB RAM, **keine GPU**, Ubuntu 24.04.5 LTS (Kernel 6.8.0-146-generic), Temurin 21.0.12.1+1 (Maven), Temurin
25.0.4.1+1 als Toolchain für `llama-ffm` (J7/J8), Maven 3.9.11, GCC 13.3.0, CMake aus Ubuntu 24.04. Branch
`ccr-90fc5f6a-vs4s0f` (`55f53d7a`), llama.cpp b11639 (`a0e6dc7384edb8f5d927155e1d26c10b4a310e6d`), statischer
CPU-Build nach der gemeinsamen Vorbereitung (`cmake -B build && cmake --build build --config Release -j16`),
Modelle aus `.github/models.csv`. Alle Läufe CPU-only (`-Dnet.ladenthin.llama.test.ngl=0`, `setDevices("none")`).
Die Proben J1–J5, J9 und F71 lagen in einer Wegwerf-Klasse `llama/src/test/java/net/ladenthin/llama/BughuntProbeTest.java`,
J7/J8 in `llama-ffm/src/test/java/net/ladenthin/llama/ffm/BughuntFfmProbeTest.java`, F93 in
`llama-langchain4j/src/test/java/…/BughuntLc4jProbeTest.java` — nur auf der VM, nicht committet, VM danach gelöscht.

---

## J1 F158: `ModelParameters.setPredict(-2)` — wie viele Tokens erzeugt der Server?

**Ziel.** Aus dem Quelltext von b11639 abgeleitet (`server-context.cpp:1964`, `:495`, `:2040`), nicht gemessen:
mit `-n -2` beendet ein Slot jede Anfrage nach dem ersten Token, weil nur `-1` als unbegrenzt gilt.

```java
try (LlamaModel model = new LlamaModel(new ModelParameters()
        .setModel(TestConstants.DRAFT_MODEL_PATH).setGpuLayers(0).setPredict(-2))) {
    CompletionResult r = model.completeWithStats(InferenceParameters.empty().withPrompt("def fib(n):"));
    System.out.println("completion_tokens=" + r.getUsage().getCompletionTokens()
            + " stop=" + r.getStopReason() + " text=" + r.getText());
}
```

Dasselbe zum Vergleich mit `setPredict(-1)` und mit `setPredict(16)`.

**Hypothese.** `-2`: 1 Token, Stopp-Grund Limit; `-1`: bis EOS oder Kontextende; `16`: 16.
**Entscheidet.** Ob `setPredict(-2)` in Java mit `IllegalArgumentException` abgelehnt werden soll.

**Ergebnis:** Probe wie im Brief (`DRAFT_MODEL_PATH` = `AMD-Llama-135m-code.Q2_K.gguf`, `completeWithStats`,
Prompt `def fib(n):`), zusätzlich die Stopp-Felder der Roh-JSON:
```
J1 setPredict(-2) completion_tokens=1 stop=MAX_TOKENS text=" Quant"
J1 setPredict(-2) raw stop fields: stop_type="limit" ... tokens_predicted=1 n_predict(gen_settings)=-2
J1 setPredict(-1) completion_tokens=2042 stop=MAX_TOKENS text="arginargs net used Net net ..."   (gekürzt)
J1 setPredict(-1) raw stop fields: stop_type="limit" ... tokens_predicted=2042 n_predict(gen_settings)=-1
J1 setPredict(16) completion_tokens=16 stop=MAX_TOKENS text=" $ NCargin Net used Net Net Net Net Net Net Net Net Net Net Net"
J1 setPredict(16) raw stop fields: stop_type="limit" ... tokens_predicted=16 n_predict(gen_settings)=16
```
(`-1`: 2042 Tokens = Kontext 2048 minus 6 Prompt-Tokens, also bis Kontextende; das Modell erreichte kein EOS.)

**Folgerung:** Die Ableitung aus dem Quelltext stimmt: `-2` beendet jede Anfrage nach **einem** Token mit
Stopp-Grund Limit, still und ohne Fehler. `setPredict(-2)` sollte in Java mit `IllegalArgumentException`
abgelehnt werden (gültig: `-1` und `>= 0`).

---

## J2 F69 / Upstream-Doku: `withNPredict(-2)` und `withNPredict(0)`

**Ziel.** (a) Bestätigen, dass der Server `n_predict: -2` pro Anfrage ablehnt (harte Grenze `-1 … INT32_MAX`,
`server-schema.cpp:44`). (b) Upstream beschreibt `0` als „no tokens will be generated but the prompt is
evaluated into the cache“; im Quelltext fand sich dafür kein eigener Zweig (`has_budget()` stoppt erst nach
dem ersten Token), deshalb steht die Aussage nicht in unserer Javadoc.

```java
// (a) erwartet: InvalidRequestException, Meldung nennt n_predict
model.complete(InferenceParameters.empty().withPrompt("x").withNPredict(-2));
// (b) Anzahl erzeugter Tokens bei 0
CompletionResult r = model.completeWithStats(InferenceParameters.empty().withPrompt("def fib(n):").withNPredict(0));
System.out.println(r.getUsage().getCompletionTokens() + " / " + r.getStopReason() + " / '" + r.getText() + "'");
```

**Entscheidet.** (a) bestätigt die korrigierte Javadoc; (b) ob die Javadoc von `withNPredict` „0 = nur den
Prompt auswerten“ nennen darf (0 Tokens) oder ob upstreams Beschreibung falsch ist (1 Token).

**Ergebnis:** ```
J2a withNPredict(-2) threw net.ladenthin.llama.exception.InvalidRequestException: Field 'n_predict': Value must be between -1 <= value <= 2147483647, but got -2
J2b withNPredict(0) 1 / MAX_TOKENS / ' args' prompt_tokens=6
J2b raw stop fields: stop_type="limit" ... tokens_predicted=1 n_predict(gen_settings)=0
```

**Folgerung:** (a) Die korrigierte Javadoc ist bestätigt: der Server lehnt `-2` pro Anfrage mit 400
(`InvalidRequestException`) und genau dieser Meldung ab. (b) `0` erzeugt **ein** Token, nicht null —
upstreams Beschreibung („no tokens will be generated“) ist für b11639 falsch. Die Javadoc von `withNPredict` darf
„0 = nur den Prompt auswerten“ **nicht** nennen; richtig ist „0 verhält sich wie 1“.

---

## J3 Offener Verdacht 5: `OpenAiCompatServer` ohne `-c` bewirbt 8192 + 2048 Tokens

**Ziel.** Ohne `-c` meldet der Server in `/props` und `/api/show` eine Kontextlänge aus den festen Vorgaben
(`DEFAULT_MAX_INPUT_TOKENS` 8192 + `DEFAULT_MAX_OUTPUT_TOKENS` 2048), unabhängig vom Modell.

```bash
java -cp "llama/target/classes:llama/src/main/natives:$(mvn -q -f llama/pom.xml dependency:build-classpath -Dmdep.outputFile=/dev/stdout)" \
  net.ladenthin.llama.server.OpenAiCompatServer -m models/AMD-Llama-135m-code.Q2_K.gguf --port 18080 &
sleep 20
curl -s localhost:18080/props | head -c 2000; echo
curl -s -X POST localhost:18080/api/show -d '{"model":"x"}' | head -c 2000; echo
kill %1
```

Dazu die tatsächliche Slot-Größe: `model.getMetricsTyped()` bzw. `/slots` → `n_ctx`, und die Trainingslänge
des Modells (`n_ctx_train` im Ladelog).

**Hypothese.** Beworben 10240, tatsächlich das Fenster des Modells (beim Draft-Modell kleiner).
**Entscheidet.** Ob der Starter das echte `n_ctx` des geladenen Modells bewerben soll.

**Ergebnis:** Befehl aus dem Brief (`OpenAiCompatServer -m models/AMD-Llama-135m-code.Q2_K.gguf --port 18080`, kein `-c`):
```
--- /props
{"default_generation_settings":{"n_ctx":10240,"model":"AMD-Llama-135m-code.Q2_K.gguf"},"total_slots":1,"model_alias":"AMD-Llama-135m-code.Q2_K.gguf","chat_template":"","modalities":{"vision":false,"audio":false,"video":false}}
--- /api/show
{"license":"","modelfile":"","parameters":"num_ctx 10240","template":"","details":{"family":"llama","families":["llama"],"parameter_size":"","quantization_level":""},"model_info":{"general.architecture":"llama","llama.context_length":10240},"capabilities":["completion","tools","insert"],"model":"AMD-Llama-135m-code.Q2_K.gguf"}
--- /slots
[{"id":0,"n_ctx":2048,"speculative":false,"is_processing":false}]
```
Ladelog: `srv    load_model: initializing, n_slots = 1, n_ctx_slot = 2048, kv_unified = 'false'`.
`getMetricsTyped()`/`getModelMeta()` am selben Modell: `J3 slot id=0 n_ctx=2048`, `"n_ctx_train":2048`.

**Folgerung:** Bestätigt: beworben werden 10240 (= 8192 + 2048), tatsächlich hat der Slot 2048 (die
Trainingslänge des Modells). Ein Client, der sich auf `/props`/`/api/show` verlässt, schickt bis zu 5× zu viel
Kontext. Der Starter sollte das echte `n_ctx` des geladenen Modells bewerben.

---

## J4 Offener Verdacht 7: liefert ein Modell leere Werkzeug-Argumente `""`?

**Ziel.** Upstream parst `function.arguments` im Verlauf mit `json::parse` (`common/chat.cpp:1026`); ein
leerer String lässt die ganze Folgeanfrage scheitern („Failed to parse tool call arguments as JSON“). Offen
ist, ob ein Modell für ein Werkzeug **ohne Parameter** `""` statt `"{}"` liefert.

Mit `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` (`--jinja`): ein Werkzeug `get_time` mit leerem Parameter-Schema
(`{"type":"object","properties":{}}`), `tool_choice: "required"`, Prompt „What time is it?“. Zehnmal
blockierend und zehnmal gestreamt (`streamChatCompletion`), jeweils `tool_calls[0].function.arguments`
wörtlich protokollieren. Danach den Verlauf mit `arguments: ""` als Assistenten-Nachricht zurückschicken
und die Antwort des Servers protokollieren (blockierend; zur Not mit `ToolCallingIntegrationTest` als Vorlage).

**Hypothese.** Das Modell liefert `"{}"`; mit `""` im Verlauf antwortet der Server mit 400.
**Entscheidet.** Ob `ParameterJsonSerializer.buildMessages` leere Argumente als `"{}"` schreiben soll.

**Ergebnis:** `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf`, `enableJinja()`, Kontext 4096. Rohe OpenAI-JSON über
`handleChatCompletions` (damit `arguments` wörtlich ankommt), Werkzeug `get_time` mit
`{"type":"object","properties":{}}`, `tool_choice: "required"`, Prompt „What time is it?“, Standard-Sampling.
```
J4 blocking#0 arguments(literal JSON string)="{}" name="get_time" finish="tool_calls"
...  (blocking#1 bis #9 identisch: "{}", get_time, tool_calls)
J4 streamed#0 arguments(concatenated)="{}" pieces=["{","}"]
...  (streamed#1 bis #9 identisch)
J4 history arguments="" -> net.ladenthin.llama.exception.InvalidRequestException: Failed to parse tool call arguments as JSON: [json.exception.parse_error.101] parse error at line 1, column 1: attempting to parse an empty input; check that your input string or stream contains the expected JSON
J4 history arguments="{}" -> OK: {"role":"assistant","content":"The current time is 12:00."}
```

**Folgerung:** Dieses Modell liefert 20/20 `"{}"`, nie `""`. Ein `""` im Verlauf lässt die ganze Folgeanfrage
aber mit 400 scheitern, `"{}"` geht durch. Der Fall ist mit diesem Modell nicht erreichbar, der Schaden bei einem
Modell oder Client, der `""` schreibt, ist jedoch total. `ParameterJsonSerializer.buildMessages` sollte leere
Argumente als `"{}"` schreiben (billig, ohne Nebenwirkung).

---

## J5 Offener Verdacht 15: wie viele Anfragen laufen bei `completeBatch` gleichzeitig?

**Ziel.** `completeBatch` (ebenso `completeBatchWithStats`, `chatBatch`) startet jede Anfrage über
`CompletableFuture.supplyAsync` auf dem gemeinsamen `ForkJoinPool` (Parallelität
`availableProcessors() - 1`), die Javadoc verspricht dagegen die mit `setParallel` eingestellte Slot-Zahl.

```java
int cores = Runtime.getRuntime().availableProcessors();
try (LlamaModel model = new LlamaModel(new ModelParameters()
        .setModel(TestConstants.DRAFT_MODEL_PATH).setGpuLayers(0).setParallel(8).setCtxSize(8 * 512))) {
    List<InferenceParameters> reqs = new ArrayList<>();
    for (int i = 0; i < 8; i++) reqs.add(InferenceParameters.empty().withPrompt("def f" + i + "(x):").withNPredict(256));
    AtomicInteger maxBusy = new AtomicInteger();
    Thread poller = new Thread(() -> { while (!Thread.currentThread().isInterrupted()) {
        maxBusy.accumulateAndGet(model.getMetricsTyped().getProcessingSlots(), Math::max);
        try { Thread.sleep(20); } catch (InterruptedException e) { return; } } });
    poller.start();
    model.completeBatch(reqs);
    poller.interrupt();
    System.out.println("cores=" + cores + " commonPool=" + ForkJoinPool.commonPool().getParallelism()
            + " maxProcessingSlots=" + maxBusy.get());
}
```

Auf einem Rechner mit 4 Kernen (bzw. mit `-Djava.util.concurrent.ForkJoinPool.common.parallelism=3`) und
auf einem mit vielen Kernen.

**Hypothese.** `maxProcessingSlots` ≈ min(8, Pool-Parallelität), also weniger als 8 bei wenigen Kernen.
**Entscheidet.** Ob `completeBatch` auf „alle im Aufrufer-Thread übergeben, dann empfangen“ umgebaut wird.

**Ergebnis:** Probe wie im Brief (Draft-Modell, `setParallel(8)`, `setCtxSize(8*512)`, 8 × 256 Tokens, Abtastung von
`getProcessingSlots()` alle 20 ms):
```
16 Kerne (Standard):                         J5 cores=16 commonPool=15 maxProcessingSlots=8 batchMillis=3225
4 Kerne (taskset -c 0-3):                    J5 cores=4 commonPool=3 maxProcessingSlots=3 batchMillis=14067
16 Kerne, -Djava.util.concurrent.ForkJoinPool.common.parallelism=3:
                                             J5 cores=16 commonPool=3 maxProcessingSlots=3 batchMillis=3262
```

**Folgerung:** Bestätigt: Die Gleichzeitigkeit ist min(Slots, Parallelität des gemeinsamen `ForkJoinPool`) — auf
4 Kernen nur 3 von 8 Slots, unabhängig von `setParallel`. Die Javadoc-Zusage stimmt nur auf Rechnern mit mehr
als `setParallel` Kernen. Umbau auf „alle im Aufrufer-Thread übergeben, dann empfangen“ empfohlen; er entkoppelt
außerdem `completeBatch` vom gemeinsamen Pool der Anwendung.

---

## J6 Modellgestützte Bestätigungen bereits gefixter Funde

Jeweils den genannten Test bzw. Ablauf laufen lassen und Zahl/Ausgabe eintragen; keiner ist Voraussetzung,
jeder bestätigt einen Fix, der ohne Modell nur bis zur Schnittstelle geprüft war.

| Fund | Lauf | Erwartung |
|---|---|---|
| F64 | `OpenAiCompatServerIntegrationTest` (Ollama-Werkzeugaufruf im Verlauf ohne `type`) | keine 400 von upstream mehr |
| F71 | `LlamaTrainerIntegrationTest` mit einem Emoji im Trainingstext (`stories260K.gguf`) | Feintuning läuft durch |
| F77 | `OpenAiCompatServerIntegrationTest` mit dem Reasoning-Modell, Anfrage mit `reasoning_effort: "none"` | kein Denkblock in der Antwort |
| F90 | zwei Runden mit Claude Codes wechselndem `cch`-Stempel im System-Prompt | `cached_tokens` > 0 in Runde 2 |
| F93 | gestreamte LangChain4j-Antwort (`test-java-llama-langchain4j-integration`) | Token-Nutzung gesetzt |
| F128 | `JllamaToolCallingIntegrationTest#streamedRequiredToolCallComesBackAsToolExecutionRequest` | grün |

```bash
mvn -B -f llama/pom.xml test -Dtest='OpenAiCompatServerIntegrationTest,LlamaTrainerIntegrationTest' -Dnet.ladenthin.llama.test.ngl=0
mvn -B -q -pl llama -am -DskipTests install && mvn -B -f llama-langchain4j/pom.xml verify \
  -Dnet.ladenthin.llama.lib.path="$PWD/llama/src/main/natives/net/ladenthin/llama/Linux/x86_64/cpu"
```

**Ergebnis:** Die beiden Befehle aus dem Brief:
```
mvn -B -f llama/pom.xml test -Dtest='OpenAiCompatServerIntegrationTest,LlamaTrainerIntegrationTest' -Dnet.ladenthin.llama.test.ngl=0
[INFO] Tests run: 12, Failures: 0, Errors: 0, Skipped: 0 -- in net.ladenthin.llama.server.OpenAiCompatServerIntegrationTest
[INFO] Tests run: 1, Failures: 0, Errors: 0, Skipped: 0 -- in net.ladenthin.llama.LlamaTrainerIntegrationTest
mvn -B -f llama-langchain4j/pom.xml verify -Dnet.ladenthin.llama.lib.path=… (+ model/embedding/rerank-Eigenschaften)
[INFO] Tests run: 55, Failures: 0, Errors: 0, Skipped: 3   (die 3 = JllamaToolCallingIntegrationTest, Eigenschaft fehlte)
[INFO] BUILD SUCCESS
```
Die vorhandenen Tests spielen die Szenarien der Tabelle nicht nach (kein Werkzeugaufruf im Ollama-Verlauf, kein
`reasoning_effort`, kein `cch`-Stempel, kein Emoji im Trainingstext). Deshalb jedes Szenario einzeln:

| Fund | Lauf | Ausgabe |
|---|---|---|
| F64 | `OpenAiCompatServer --jinja` mit Qwen2.5-1.5B, `POST /api/chat`, Verlauf `assistant.tool_calls=[{"function":{"name":"get_time","arguments":{}}}]` (ohne `type`) + `tool`-Antwort | `HTTP 200`, `{"message":{"role":"assistant","content":"The current time is 12:00."},…,"done":true,"done_reason":"stop"}` |
| F71 | Feintuning wie `LlamaTrainerIntegrationTest` (`stories260K.gguf`, nCtx 128, ADAMW), Korpus mit 🦊 und 🐶 (128 Zeichen außerhalb der BMP) | `J6/F71 finetune OK: input=1185376 B, output=1201728 B, weights changed=true` |
| F77 | `OpenAiCompatServer --jinja` mit Qwen3-0.6B, `/v1/chat/completions` „What is 2+2?“ mit `reasoning_effort: "none"` gegen ohne | mit `none`: `reasoning_content_len=0 think_in_content=False completion_tokens=9 content='2 + 2 = 4.'`; ohne: `reasoning_content_len=459 completion_tokens=133` |
| F90 | `/v1/messages` (Anthropic), System-Prompt `x-anthropic-billing-header: cc_version=2.1.3.abc; cc_entrypoint=cli; cch=…;` + ~880 Token, drei Runden | `cch=a5145`: `cache_read_input_tokens 3`; **`cch=b7731`: `input_tokens 1, cache_read_input_tokens 885`**; Gegenprobe mit geändertem `cc_version`: `cache_read_input_tokens 18` |
| F93 | `JllamaStreamingChatModel` und `JllamaChatModel`, Qwen2.5-1.5B, „Reply with the single word: ok“ | gestreamt: `tokenUsage=TokenUsage { inputTokenCount = 36, outputTokenCount = 2, totalTokenCount = 38 } finishReason=STOP`; blockierend identisch |
| F128 | `JllamaToolCallingIntegrationTest` mit `-Dnet.ladenthin.llama.langchain4j.tool.model=…/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` | `Tests run: 3, Failures: 0, Errors: 0, Skipped: 0` (enthält `streamedRequiredToolCallComesBackAsToolExecutionRequest`) |

**Folgerung:** Alle sechs Fixes sind mit Modell bestätigt: F64 kein 400 mehr, F71 Feintuning läuft mit Emoji
durch und ändert die Gewichte, F77 unterdrückt den Denkblock (9 statt 133 Tokens), F90 trifft den Präfix-Cache
in Runde 2 (885 von 886 Tokens aus dem Cache; die Gegenprobe zeigt, dass ohne die Normalisierung nur der
Anfang vor der Änderung getroffen würde), F93 setzt die Token-Nutzung im Stream, F128 grün. Hinweis für CI: der
Integrationsjob sollte `net.ladenthin.llama.langchain4j.tool.model` setzen, sonst bleibt F128 dort übersprungen.

---

## J7 F59: `FfmNativeServer.isRunning()` gegen ein paralleles `close()` (JDK 22+)

**Ziel.** `isRunning()` kann nach einem parallelen `close()` freigegebenen Speicher lesen. Vorgeschlagener
Rot-Test: paketinterne Naht für die beiden Method-Handles, `isRunning` in einem Latch festhalten, `close()`
parallel; Erwartung nach dem Fix: `close()` wartet. Lauf im Modul `llama-ffm` (JDK 22+, wie Job
`test-java-llama-ffm`): `mvn -B -f llama-ffm/pom.xml verify -Dnet.ladenthin.llama.lib.path=…`.

**Entscheidet.** Ob das Lock aus `NativeServer` übernommen wird (rot = ja).

**Ergebnis:** `mvn -B -f llama-ffm/pom.xml test -Dtest=BughuntFfmProbeTest#… -Dnet.ladenthin.llama.lib.path=…`
(Toolchain: `Found matching toolchain for type jdk: JDK[/opt/jdk-25.0.4.1+1]`).

1. **Rot-Test mit paketinterner Naht** (nur auf der VM): `FfmNativeServer` bekam zwei statische Felder
   `isRunningHandle`/`stopHandle` (vorbelegt mit `SERVER_IS_RUNNING`/`SERVER_STOP`) statt der direkten Verweise.
   Der Test ersetzt sie durch Java-Handles: `isRunning` hält in einem Latch fest, `close()` läuft parallel.
   ```
   J7 seam events: [isRunning:native-entered, close:native-stop (frees the handle), close() RETURNED after 2 s, isRunning:native-returning]
   J7 seam verdict: close() did NOT wait for the isRunning() native call (RED: stop/free happened while it was inside)
   Tests run: 1, Failures: 1   (AssertionError "J7 RED: close() returned while isRunning() was inside the native call")
   ```
2. **Ohne Naht, echter Server** (Router-Modus über ein leeres `--models-dir`, 30 Runden, ein Thread ruft
   `isRunning()` in einer Schleife, der Haupt-Thread `close()`):
   ```
   J7 real: rounds=30 isRunning calls=862476291 true-after-close=0 (JVM survived)
   ```
Quelltext des Rot-Tests (Auszug; Naht: `static MethodHandle isRunningHandle = JllamaNative.SERVER_IS_RUNNING;`,
`static MethodHandle stopHandle = JllamaNative.SERVER_STOP;`, in `isRunning()`/`close()` statt der Konstanten):
```java
static boolean fakeIsRunning(MemorySegment handle) throws InterruptedException {
    EVENTS.add("isRunning:native-entered"); ENTERED.countDown(); RELEASE.await();
    EVENTS.add("isRunning:native-returning"); return true;
}
static int fakeStop(MemorySegment handle) { EVENTS.add("close:native-stop (frees the handle)"); return 0; }

@Test public void j7Seam() throws Exception {
    MethodHandles.Lookup l = MethodHandles.lookup();
    FfmNativeServer.isRunningHandle = l.findStatic(BughuntFfmProbeTest.class, "fakeIsRunning",
            MethodType.methodType(boolean.class, MemorySegment.class));
    FfmNativeServer.stopHandle = l.findStatic(BughuntFfmProbeTest.class, "fakeStop",
            MethodType.methodType(int.class, MemorySegment.class));
    Constructor<FfmNativeServer> c = FfmNativeServer.class.getDeclaredConstructor(MemorySegment.class);
    c.setAccessible(true);
    FfmNativeServer server = c.newInstance(MemorySegment.NULL);
    Thread reader = new Thread(server::isRunning); reader.start(); ENTERED.await();
    Thread closer = new Thread(server::close); closer.start(); closer.join(2000);
    boolean closeReturnedWhileIsRunningInside = !closer.isAlive();
    RELEASE.countDown(); reader.join(); closer.join();
    if (closeReturnedWhileIsRunningInside) throw new AssertionError("J7 RED: close() returned while isRunning() was inside the native call");
}
```

**Folgerung:** Rot: `close()` gibt den Handle frei, während `isRunning()` noch im nativen Aufruf auf ihm arbeitet.
Das Fenster ist im Normalbetrieb schmal (862 Mio. Aufrufe ohne sichtbaren Schaden), ein Use-after-free bleibt
es trotzdem. Das Lock aus `NativeServer` sollte übernommen werden (nach dem Fix muss der Test grün werden:
`close()` wartet).

---

## J8 Offener Verdacht 1: `JllamaNative.takeStrings` gibt bei `count == 0` nicht frei (JDK 22+)

**Ziel.** Bei `count == 0` kehrt `takeStrings` vor dem `try` zurück, `STRINGS_FREE` wird für den
Nicht-NULL-Zeiger nie gerufen (8 Byte je Aufruf; praktisch unerreichbar). Vorgeschlagener Test:
`takeStrings` mit der Freigabefunktion als Parameter, Aufruf mit Nicht-NULL-Zeiger und Anzahl 0, Erwartung
genau ein Freigabeaufruf.

**Entscheidet.** Ob der Fix („bei Nicht-NULL immer freigeben“) umgesetzt wird.

**Ergebnis:** `takeStrings` hat keine Naht für die Freigabefunktion (`STRINGS_FREE` ist `static final`), der
vorgeschlagene Test bräuchte also eine Produktionsänderung. Stattdessen deterministisch über glibc: ein
**falsch ausgerichteter** Nicht-NULL-Zeiger, auf dem `free()` sofort mit `free(): invalid pointer` abbricht.
```
mode=control      (jllama_strings_free(p, 0) direkt):  mvn exit=1  Process Exit Code: 134   free(): invalid pointer
mode=takeStrings  (JllamaNative.takeStrings(p, 0)):    mvn exit=0  Tests run: 1, Failures: 0
  J8 mode=takeStrings: calling with misaligned non-NULL pointer 0x755a0072b991 and count 0
  J8 mode=takeStrings: returned [] and the JVM survived -> the free function was NOT called
```
Probe:
```java
@Test public void j8InvalidPointer() throws Throwable {
    String mode = System.getProperty("j8.mode", "takeStrings");
    try (Arena arena = Arena.ofConfined()) {
        MemorySegment p = MemorySegment.ofAddress(arena.allocate(64, 16).address() + 1);
        MemorySegment out = arena.allocate(ADDRESS); MemorySegment len = arena.allocate(JAVA_LONG);
        out.set(ADDRESS, 0, p); len.set(JAVA_LONG, 0, 0L);
        if (mode.equals("takeStrings")) JllamaNative.takeStrings(out, len);
        else JllamaNative.invoke(JllamaNative.STRINGS_FREE, p, 0L);
    }
}
```
(Zwei vorherige Versuche über den Heap-Zähler `mallinfo2().uordblks` und über die tcache-Double-Free-Erkennung
waren nicht aussagekräftig — die JVM allokiert dazwischen selbst, die Gegenprobe schlug nicht an — und sind
deshalb nicht Teil des Ergebnisses.)

**Folgerung:** Bestätigt: Bei `count == 0` ruft `takeStrings` die Freigabe nie auf, ein Nicht-NULL-Array leckt
(laut Verdacht 8 Byte je Aufruf); `jllama_strings_free(p, 0)` würde es freigeben. Der Fix „bei Nicht-NULL immer
freigeben“ ist richtig und gefahrlos.

---

## J9 Nur falls das Verhalten geändert werden soll: F67 (LoRA-Cache)

Braucht ein kleines Modell mit passendem LoRA-Adapter (nicht in `models.csv`). Ablauf: denselben Prompt
mit Greedy-Sampling vor und nach `setLoraAdapter(0, 0f)` senden und mit einem frisch geladenen Modell ohne
Adapter vergleichen; dann dasselbe nach `eraseSlot(0)`. Ohne Adapter: als „nicht ausführbar“ beantworten.

**Entscheidet.** Ob `engine::set_lora_adapters` nach dem Wechsel die Slots leeren soll.

**Ergebnis:** Kein Adapter in `models.csv`; genutzt wurde das Paar aus llama.cpps eigenen Server-Tests
(`tools/server/tests/unit/test_lora.py`): `ggml-org/stories15M_MOE` → `stories15M_MOE-F16.gguf` mit
`moe_shakespeare15M.gguf`. Greedy (`temperature 0, top_k 1, seed 42`, `cache_prompt true`, Slot 0, `setParallel(1)`,
Kontext 512), Prompt „Look in thy glass“, 32 Tokens:
```
J9 fresh model, no adapter:            " was a little girl. She was only three years old and she was three years old. She was three years old. She was three years old and she was"
J9 adapters at load: [LoraAdapter(id=0, path=…/moe_shakespeare15M.gguf, scale=1.0, …)]
J9 adapter scale 1 (cold slot):        " in love of the eye:\nThat's when when the eye see thy on the sun's shining bright.\nThat's why those flowers"
J9 adapter scale 1 (warm slot):        " in love of the eye:\nThat's when when the eye see thy on the sun's shining bright.\nThat's why those flowers"
J9 adapters after setLoraAdapter(0,0): [LoraAdapter(id=0, …, scale=0.0, …)]
J9 scale 0, warm slot (cache kept):    ". He saw a picture of a bird and he wanted to know what it was. He asked his mom, \"What is it?\"\n\"It's" equalsFresh=false
J9 eraseSlot(0): {"id_slot":0,"n_erased":36}
J9 scale 0 after eraseSlot(0):         " was a little girl. She was only three years old and she was three years old. She was three years old. She was three years old and she was" equalsFresh=true
```

**Folgerung:** F67 ist bestätigt: Nach `setLoraAdapter(0, 0f)` rechnet der Slot mit dem unter Skala 1
berechneten Prompt-Cache weiter und liefert eine dritte, weder dem Adapter noch dem Grundmodell entsprechende
Ausgabe. Erst `eraseSlot(0)` stellt das Verhalten eines frischen Modells her. `engine::set_lora_adapters` sollte
nach einem Wechsel die Slots leeren, wenn das Verhalten geändert werden soll.
