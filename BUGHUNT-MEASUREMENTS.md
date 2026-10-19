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

**Ergebnis:**

**Folgerung:**

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

**Ergebnis:**

**Folgerung:**

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

**Ergebnis:**

**Folgerung:**

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

**Ergebnis:**

**Folgerung:**

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

**Ergebnis:**

**Folgerung:**

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

**Ergebnis:**

**Folgerung:**

---

## J7 F59: `FfmNativeServer.isRunning()` gegen ein paralleles `close()` (JDK 22+)

**Ziel.** `isRunning()` kann nach einem parallelen `close()` freigegebenen Speicher lesen. Vorgeschlagener
Rot-Test: paketinterne Naht für die beiden Method-Handles, `isRunning` in einem Latch festhalten, `close()`
parallel; Erwartung nach dem Fix: `close()` wartet. Lauf im Modul `llama-ffm` (JDK 22+, wie Job
`test-java-llama-ffm`): `mvn -B -f llama-ffm/pom.xml verify -Dnet.ladenthin.llama.lib.path=…`.

**Entscheidet.** Ob das Lock aus `NativeServer` übernommen wird (rot = ja).

**Ergebnis:**

**Folgerung:**

---

## J8 Offener Verdacht 1: `JllamaNative.takeStrings` gibt bei `count == 0` nicht frei (JDK 22+)

**Ziel.** Bei `count == 0` kehrt `takeStrings` vor dem `try` zurück, `STRINGS_FREE` wird für den
Nicht-NULL-Zeiger nie gerufen (8 Byte je Aufruf; praktisch unerreichbar). Vorgeschlagener Test:
`takeStrings` mit der Freigabefunktion als Parameter, Aufruf mit Nicht-NULL-Zeiger und Anzahl 0, Erwartung
genau ein Freigabeaufruf.

**Entscheidet.** Ob der Fix („bei Nicht-NULL immer freigeben“) umgesetzt wird.

**Ergebnis:**

**Folgerung:**

---

## J9 Nur falls das Verhalten geändert werden soll: F67 (LoRA-Cache)

Braucht ein kleines Modell mit passendem LoRA-Adapter (nicht in `models.csv`). Ablauf: denselben Prompt
mit Greedy-Sampling vor und nach `setLoraAdapter(0, 0f)` senden und mit einem frisch geladenen Modell ohne
Adapter vergleichen; dann dasselbe nach `eraseSlot(0)`. Ohne Adapter: als „nicht ausführbar“ beantworten.

**Entscheidet.** Ob `engine::set_lora_adapters` nach dem Wechsel die Slots leeren soll.

**Ergebnis:**

**Folgerung:**
