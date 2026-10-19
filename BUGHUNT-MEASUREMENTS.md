<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT
-->

# Messauftrag zu F67 und F179 … F186 — java-llama.cpp

Die Fixes vom 2026-10-19/20 sind alle modellfrei rot/grün belegt: C++ über ein Zufallsmodell, Java über
Fake-Backends oder ein lokal erzeugtes Zufalls-GGUF. Was ein Zufallsmodell nicht zeigen kann — echte
Texte, echte Stoppwörter, echte Wahrscheinlichkeiten, echte Adapter, Windows-Dateinamen — sammelt dieser
Auftrag. Hintergrund in `BUGHUNT-FOUND.md` (die Fund-IDs) und `CHANGELOG.md` („Fixed“).

**Regeln für den ausführenden Agenten**

- **Nur messen und berichten.** Keine Produktionsänderung, keine Fixes. Wegwerf-Proben (Testklassen,
  kleine `main`-Programme) nach dem Lauf entfernen; committet wird nur diese Datei mit den Ergebnissen.
- Arbeitsstand: Branch `ccr-90fc5f6a-vs4s0f` (mindestens `13ddf17`). Pin: llama.cpp b11639.
- Dateien aus `.github/shared-files.sha256` nicht verändern.
- Jeder Punkt wird beantwortet, auch wenn er nicht ausführbar war (mit Grund). Zu jedem Ergebnis die
  Umgebung (CPU, Kerne, RAM, GPU, OS, JDK) und die wörtlichen Ausgaben, die das Ergebnis tragen.
- Ergebnis unter **Ergebnis:**, darunter **Folgerung:** (bestätigt der Lauf den Fix, oder was weicht ab).
- Ein Test, der hier rot wird, ist ein Befund: Ausgabe vollständig eintragen, nicht „reparieren“.

**Gemeinsame Vorbereitung**

```bash
# JDK: Temurin 21 (nicht Oracle 21, siehe CLAUDE.md); für M10 zusätzlich ein JDK 22+ in ~/.m2/toolchains.xml
mvn -B -q -f llama/pom.xml compile
cd llama && cmake -B build -DBUILD_TESTING=ON && cmake --build build --config Release -j"$(nproc)" && cd ..
ctest --test-dir llama/build --output-on-failure        # erwartet: 643/643
# Modelle aus .github/models.csv nach models/ (die Tests finden sie dort):
mkdir -p models
while IFS=, read -r f url; do case "$f" in \#*|"") continue;; esac
  [ -f "models/$f" ] || curl -L --fail -o "models/$f" "$url"; done < .github/models.csv
# Auf einem Rechner ohne GPU alle Läufe mit -Dnet.ladenthin.llama.test.ngl=0
```

Für die kleinen Messprogramme unten genügt eine Wegwerf-Testklasse unter
`llama/src/test/java/net/ladenthin/llama/` (dann gilt der Test-Klassenpfad mit `src/main/natives/`),
gestartet mit `mvn -B -q -f llama/pom.xml test -Dtest=<Klasse>`. Pfade über `TestConstants`
(`MODEL_PATH`, `DRAFT_MODEL_PATH`, `REASONING_MODEL_PATH`, `DEFAULT_TOOL_MODEL_PATH`).

---

## M1 Die neuen und geänderten modellgestützten Tests

**Ziel.** Die Tests, die nur in CI mit Modellen laufen, einmal lokal mit dem echten Modellsatz grün sehen.

```bash
mvn -B -f llama/pom.xml test -Dnet.ladenthin.llama.test.ngl=0 -Dtest='ErrorHandlingTest,OpenAiServerCompletionIntegrationTest,OpenAiServerRerankIntegrationTest,OpenAiCompatServerIntegrationTest,LlamaTrainerTest,LlamaTrainerIntegrationTest'
```

Erwartet: alle grün, keiner übersprungen. Im Einzelnen:
`ErrorHandlingTest#testNPredictMinusTwoPerRequestIsAnInvalidRequest` (F69: `n_predict: -2` pro Anfrage
ist `InvalidRequestException`); `OpenAiServerCompletionIntegrationTest#aStreamedCompletionIsUpstreamsStream`
(F186), `#ollamaGenerateRawRoundTrip` und `#ollamaGenerateNonStreamingRoundTrip` (F179);
`OpenAiServerRerankIntegrationTest` (F177).

**Ergebnis:**

**Folgerung:**

---

## M2 F186: das gestreamte `/v1/completions` mit dem echten Modell

**Ziel.** Der Integrationstest prüft die Form; hier zusätzlich der Inhalt. Mit `OpenAiCompatServer` über
`MODEL_PATH` (CodeLlama) dieselbe Anfrage einmal blockierend und einmal gestreamt, Greedy:

```json
{"prompt": "def fibonacci(n):", "max_tokens": 24, "temperature": 0, "top_k": 1, "logprobs": 3}
```

Aufzeichnen: (a) der zusammengesetzte Text aller gestreamten `choices[0].text` gegen `choices[0].text`
der blockierenden Antwort; (b) `usage` des letzten Chunks gegen `usage` der blockierenden Antwort;
(c) die ersten drei `logprobs.content`-Einträge beider Wege; (d) dasselbe mit `"prompt": [<die Token-Ids
von "def fibonacci(n):">]` (Ids über `LlamaModel.encode`); (e) `"prompt": ["a", "b"]` gestreamt und
blockierend (erwartet: beide `400`, Meldung nennt „2 prompts“, F184).

**Hypothese.** (a) gleich, (b) gleiche Zahlen, (c) gleiche Tokens und Werte, (d) wie (a), (e) zweimal 400.

**Ergebnis:**

**Folgerung:**

---

## M3 F183: Anthropic `stop_sequence` mit einem echten Stoppwort

**Ziel.** Mit dem Tool-Modell (`DEFAULT_TOOL_MODEL_PATH`, Qwen2.5-1.5B) über `OpenAiCompatServer`:

```json
{"model": "m", "max_tokens": 64, "temperature": 0, "stop_sequences": ["5"],
 "messages": [{"role": "user", "content": "Count from 1 to 10, separated by commas."}]}
```

an `/v1/messages`, einmal blockierend, einmal mit `"stream": true`. Aufzeichnen: `stop_reason`,
`stop_sequence`, der Text; im Stream das `message_delta`-Ereignis. Dazu dieselbe Anfrage ohne
`stop_sequences` (erwartet `stop_sequence: null`) und eine, die an `max_tokens` endet (`max_tokens: 3`,
erwartet `stop_reason: max_tokens`, `stop_sequence: null`). Prüfen, dass `__verbose` in keiner Antwort und
keinem Ereignis auftaucht.

**Hypothese.** `end_turn` mit `stop_sequence: "5"`, Text endet vor der 5; die beiden Gegenproben wie erwartet.

**Ergebnis:**

**Folgerung:**

---

## M4 F182: ein gestreamter Fehler vor dem ersten Chunk mit dem echten Modell

**Ziel.** Ein Prompt über der Kontextgröße ist bei upstream `400` (`exceed_context_size_error`), gestreamt
wie nicht gestreamt. Mit `OpenAiCompatServer` über `DRAFT_MODEL_PATH` und `setCtxSize(256)` eine
Chat-Anfrage mit ~2000 Wörtern Inhalt an alle fünf Streaming-Routen (`/v1/chat/completions`,
`/v1/completions`, `/api/chat`, `/v1/messages`, `/v1/responses`, jeweils mit `"stream": true`).
Aufzeichnen: HTTP-Status, `Content-Type`, Körper.

**Hypothese.** Überall `400` mit JSON-Körper im Fehlerformat der Route, kein SSE.

**Ergebnis:**

**Folgerung:**

---

## M5 F185: `id` und `model` über beide Server

**Ziel.** (a) `OpenAiCompatServer` mit `--model-id my-model` (Kommandozeile über `ServerLauncher
--jllama-openai-compat`): `/v1/models`, eine blockierende Chat-Antwort und ein Stream; jedes `id` beginnt
mit `chatcmpl-`, alle Chunks eines Streams tragen dieselbe, `model` ist überall `my-model`. (b) Dasselbe
ohne `--model-id` (erwartet: der Dateiname). (c) `NativeServer` mit `--alias my-model` als Gegenprobe gegen
upstreams eigenen Server im selben Prozess.

**Ergebnis:**

**Folgerung:**

---

## M6 F67 / Patch 0021: LoRA-Wechsel mit dem echten Adapterpaar

**Ziel.** J9 des vorigen Auftrags mit dem Patch wiederholen. Paar aus llama.cpps Server-Tests:
`ggml-org/stories15M_MOE` → `stories15M_MOE-F16.gguf` und `moe_shakespeare15M.gguf`. Greedy
(`temperature 0, top_k 1, seed 42`, `cache_prompt true`, Slot 0, `setParallel(1)`, Kontext 512), Prompt
„Look in thy glass“, 32 Tokens. Zwei Durchgänge: mit `setCacheRamMib(0)` und mit der Voreinstellung.

Ablauf je Durchgang: frisches Modell ohne Adapter (Referenz A); mit Adapter Skala 1, kalt und warm
(Referenz B); `setLoraAdapter(0, 0f)` und sofort dieselbe Anfrage auf dem warmen Slot; dann
`setLoraAdapter(0, 1f)` und wieder dieselbe Anfrage; dann eine Anfrage mit eigenem `"lora": [{"id": 0,
"scale": 0}]`, gefolgt von einer ohne.

**Hypothese.** Mit `setCacheRamMib(0)`: nach `0f` gleich A, nach `1f` gleich B, nach der Anfrage mit eigenen
Adaptern die nächste gleich B. Mit der Voreinstellung kann der Host-Speicher-Cache (Javadoc von
`setLoraAdapters`) abweichen — das ist der offene Rest, genau aufzeichnen.

**Ergebnis:**

**Folgerung:**

---

## M7 F181: Fehlermeldungen mit einem 4-Byte-Zeichen auf Windows

**Ziel.** `LlamaTrainerTest` läuft in CI auf Linux; der Windows-Weg (UTF-16-Dateinamen, `cmd`-Codepage)
ist ungeprüft. Auf Windows: `mvn -B -f llama/pom.xml test -Dtest=LlamaTrainerTest`, dann eine Wegwerf-Probe,
die `new LlamaModel(new ModelParameters().setModel("C:/tmp/fehlt-\uD83D\uDE00.gguf"))` und
`NativeServer` mit demselben Pfad startet und die Meldung codepunktweise ausgibt.

**Hypothese.** Beide Tests grün (nicht übersprungen); die Probe zeigt U+1F600 als einen Codepunkt, nichts
abgeschnitten.

**Ergebnis:**

**Folgerung:**

---

## M8 F182 im Agenten: Atmosphere über das echte Modell

**Ziel.** Seit F182 meldet der Agent einen Fehler vor dem ersten Chunk als Fehler des Zuges statt still zu
enden. Lauf: `mvn -B -f llama-atmosphere-agent/pom.xml -Dllama.version=6.0.0-SNAPSHOT -Dllama.natives=none
verify` gegen den installierten Kern (`mvn -pl llama,llama-platform -am install -DskipTests`), dann
`AtmosphereToolLoopIntegrationTest` mit dem Tool-Modell (`-Dnet.ladenthin.llama.lib.path=<natives-Verzeichnis>`).
Dazu ein Zug im Agenten mit einem Prompt über der Kontextgröße (`--ctx-size 256`): was zeigt die Konsole?

**Ergebnis:**

**Folgerung:**

---

## M9 `llama-ffm` mit dem echten Modell

**Ziel.** `mvn -B -f llama-ffm/pom.xml verify -Dnet.ladenthin.llama.ffm.model=models/AMD-Llama-135m-code.Q2_K.gguf
-Dnet.ladenthin.llama.lib.path=<…/llama/src/main/natives/net/ladenthin/llama/<OS>/<ARCH>/cpu>` — neu ist
`FfmLlamaModelIntegrationTest#oaiCompletionStreamEndsWithUsage` (F186).

**Ergebnis:**

**Folgerung:**

---

## M10 Gesamtlauf

**Ziel.** Nach M1 … M9 einmal alles: `ctest` (erwartet 643/643) und
`mvn -B -f llama/pom.xml test -Dnet.ladenthin.llama.test.ngl=0` mit dem ganzen Modellsatz, dazu
`.github/verify-test-counts.sh` wie in CI. Zahlen (run / failures / errors / skipped) eintragen.

**Ergebnis:**

**Folgerung:**
