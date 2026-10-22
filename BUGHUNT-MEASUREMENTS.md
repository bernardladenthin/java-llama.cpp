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
- Arbeitsstand: Branch `ccr-90fc5f6a-vs4s0f` (mindestens `573619a`). Pin: llama.cpp b11639.
- Dateien aus `.github/shared-files.sha256` nicht verändern.
- Jeder Punkt wird beantwortet, auch wenn er nicht ausführbar war (mit Grund). Zu jedem Ergebnis die
  Umgebung (CPU, Kerne, RAM, GPU, OS, JDK) und die wörtlichen Ausgaben, die das Ergebnis tragen.
- Ergebnis unter **Ergebnis:**, darunter **Folgerung:** (bestätigt der Lauf den Fix, oder was weicht ab).
- Ein Test, der hier rot wird, ist ein Befund: Ausgabe vollständig eintragen, nicht „reparieren“.

**Eine Ausnahme von der ersten Regel, ausdrücklich beauftragt:** die Befunde dieses Auftrags wurden
nach Abschluss der Messungen behoben, jeder mit einem Wächter und jeder end-to-end belegt — **B1**
(M6, `--lora-scaled` rendert das falsche Werteformat) samt dem dabei gefundenen **B4** (die vier
`addX`-Methoden ersetzten statt anzuhängen), **B2** (M5, `/v1/models` und die Antworten benennen das
Modell verschieden) und **B3** (M8, der Testharnisch schreibt unescapte Steuerzeichen in sein JSON).
Die Messabschnitte selbst stehen unverändert so da, wie sie gemessen wurden; was danach geändert
wurde, steht jeweils in einem eigenen Unterabschnitt „ist behoben". Nicht angefasst sind die
Nebenbeobachtungen aus M7 und der eingebettete Weg aus B2 — bei beiden ist begründet, warum.

Die vier Befunde stehen mit ihren globalen Fund-IDs in [`BUGHUNT-FOUND.md`](BUGHUNT-FOUND.md): **F187** (B1), **F188** (B4), **F189** (B2) und **F190** (B3).

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

## Umgebung aller Läufe (einmal, gilt für M1 … M10)

| | |
|---|---|
| CPU | AMD Ryzen 7 5800H with Radeon Graphics, 8 Kerne / 16 Threads |
| RAM | 63,3 GiB |
| GPU | NVIDIA GeForce RTX 3070 Laptop (Treiber 32.0.15.8183) + AMD Radeon (iGPU) — **ungenutzt**, siehe unten |
| OS | Microsoft Windows 11 Pro 10.0.26300 (Build 26300.9711) |
| JDK (Maven, Kern) | Temurin 21.0.12.1+1 — **nicht** das daneben installierte Oracle 21.0.9, das `mvn compile` laut CLAUDE.md in Error Prone abbricht |
| JDK (`llama-ffm`) | Temurin 25.0.4.1+1, über einen neu angelegten `~/.m2/toolchains.xml` |
| Maven | 3.9.11 |
| C++-Toolchain | MSVC 19.44.35229.0 (`cl.exe`), Ninja Multi-Config, CMake 3.31.6-msvc6 aus VS 2022 Community |
| Build | statisch mit `-DBUILD_TESTING=ON` (`.github\build.bat`), **ohne** sccache (`USE_CACHE=false`) |

**Der Build ist CPU-only.** Es wurde kein GPU-Modul gebaut, das Natives-Verzeichnis trägt genau
`Windows/x86_64/cpu/`, und `JNI_OnLoad` meldet in jedem Lauf
`[jllama] native backend 'cpu' loaded from … with no GPU module`. Alle Läufe daher mit
`-Dnet.ladenthin.llama.test.ngl=0`; die RTX 3070 ist an keiner Messung beteiligt.

`llama/src/main/natives/net/ladenthin/llama/Windows/x86_64/cpu/jllama-build.txt` des gemessenen Builds:

```
llama.cpp=b11639
llama.cpp.commit=a0e6dc7384edb8f5d927155e1d26c10b4a310e6d
backend=cpu
kind=library
classifier=cpu-windows-x86-64
jllama.version=6.0.0-SNAPSHOT
jllama.commit=86d0f8a8746feb4accc998c67c80e44ffd63874f
patches=0001,0002,0003,0006,0007,0008,0014,0015,0016,0017,0018,0019,0020,0021
patches.sha256=0176a9b12e147b97027b393ca107f559bc9c997f6076b5a94390957bf10106b8
compiler=MSVC 19.44.35229.0
toolkit=none
signed=no
```

Arbeitsstand also `86d0f8a8` auf `ccr-90fc5f6a-vs4s0f` (die verlangte Mindesthöhe `573619a` ist
enthalten), Pin b11639, alle 14 Patches angewandt.

**Modelle.** Die elf Zeilen aus `.github/models.csv` lagen bereits vollständig in `models/`; nichts
wurde nachgeladen. Für M6 kamen `stories15M_MOE-F16.gguf` (73 466 432 B) und
`moe_shakespeare15M.gguf` (16 364 896 B) von `ggml-org/stories15M_MOE` hinzu, beide mit GGUF-Magic
`47475546`.

**Vorbereitung.** `mvn -f llama/pom.xml compile` grün; der Build lief 549/549 Schritte durch; danach

```
100% tests passed, 0 tests failed out of 643
Total Test time (real) = 150.90 sec
```

also die vom Auftrag erwarteten **643/643**.

Ein Hinweis zum Ausgangszustand: die vorhandenen Natives waren `llama.cpp=b11538` vom 11.10. und
damit älter als Pin und Patches `0020`/`0021`. `llama/build` und `llama/src/main/natives` wurden
deshalb gelöscht und alles frisch gebaut.

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

**Ergebnis:** Alle sechs Klassen grün, **keine übersprungen**:

```
[INFO] Running net.ladenthin.llama.ErrorHandlingTest
[INFO] Tests run: 17, Failures: 0, Errors: 0, Skipped: 0, Time elapsed: 7.544 s
[INFO] Running net.ladenthin.llama.LlamaTrainerIntegrationTest
[INFO] Tests run: 1, Failures: 0, Errors: 0, Skipped: 0, Time elapsed: 0.793 s
[INFO] Running net.ladenthin.llama.LlamaTrainerTest
[INFO] Tests run: 2, Failures: 0, Errors: 0, Skipped: 0, Time elapsed: 0.021 s
[INFO] Running net.ladenthin.llama.server.OpenAiCompatServerIntegrationTest
[INFO] Tests run: 12, Failures: 0, Errors: 0, Skipped: 0, Time elapsed: 6.583 s
[INFO] Running net.ladenthin.llama.server.OpenAiServerCompletionIntegrationTest
[INFO] Tests run: 7, Failures: 0, Errors: 0, Skipped: 0, Time elapsed: 12.78 s
[INFO] Running net.ladenthin.llama.server.OpenAiServerRerankIntegrationTest
[INFO] Tests run: 1, Failures: 0, Errors: 0, Skipped: 0, Time elapsed: 1.303 s
[INFO] Tests run: 40, Failures: 0, Errors: 0, Skipped: 0
[INFO] BUILD SUCCESS
```

Weil eine Klasse mit einer `@BeforeAll`-Annahme als `tests="0"` durchläuft und damit unsichtbar
bliebe (CLAUDE.md, „CI model policy"), sind die namentlich genannten Methoden zusätzlich einzeln aus
den surefire-XML belegt — jede mit einer eigenen `<testcase>`-Zeile und einer Laufzeit > 0:

```
name="testNPredictMinusTwoPerRequestIsAnInvalidRequest" classname="net.ladenthin.llama.ErrorHandlingTest" time="0.06"
name="aStreamedCompletionIsUpstreamsStream" classname="…OpenAiServerCompletionIntegrationTest" time="0.438"
name="ollamaGenerateRawRoundTrip"           classname="…OpenAiServerCompletionIntegrationTest" time="2.106"
name="ollamaGenerateNonStreamingRoundTrip"  classname="…OpenAiServerCompletionIntegrationTest" time="2.881"
testcase name="rerankReturnsScoredResultsCappedByTopN"                       (OpenAiServerRerankIntegrationTest)
testcase name="aFourByteCharacterInAPathComesBackIntactInTheErrorMessage"    (LlamaTrainerTest)
testcase name="aFourByteCharacterInTheTrainingTextReachesTheTrainer"         (LlamaTrainerTest)
testcase name="finetuneWritesAnOutputModel"                                  (LlamaTrainerIntegrationTest)
```

**Folgerung:** Der Lauf bestätigt die Erwartung. F69, F186, F179 und F177 sind mit dem echten
Modellsatz auf Windows belegt, und `LlamaTrainerTest` — in CI nur auf Linux — läuft auch hier grün,
was die erste Hälfte von M7 bereits mit abdeckt.

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

**Ergebnis:** Der erste Durchgang bestätigte (a), (d) und (e) sofort, zeigte aber in (b) und (c)
Abweichungen — die sich als **Prompt-Cache** erklärten: das Paar läuft blockierend zuerst, die
gestreamte Anfrage trifft denselben Slot warm. Deshalb lief dasselbe Paar ein zweites Mal mit
`"cache_prompt": false`. Beide Durchgänge unten.

**(a) Text, mit Prompt-Cache** — identisch, blockierend wie gestreamt (25 Chunks):

```
  (a) blocking text = [
  if n == 0:
    return 0
  elif n == 1:
    return]
  (a) streamed text = [… byte-identisch …]
  (a) EQUAL=true
```

**(b) mit Prompt-Cache** — die Zahlen gleich, nur `cached_tokens` unterschiedlich:

```
  (b) blocking usage   = {"completion_tokens":24,"prompt_tokens":8,"total_tokens":32,"prompt_tokens_details":{"cached_tokens":0}}
  (b) last-chunk usage = {"completion_tokens":24,"prompt_tokens":8,"total_tokens":32,"prompt_tokens_details":{"cached_tokens":7}}
  (b) EQUAL=false
```

**(b) Gegenprobe mit `cache_prompt:false`** — vollständig gleich:

```
  (b) blocking usage   = {"completion_tokens":24,"prompt_tokens":8,"total_tokens":32,"prompt_tokens_details":{"cached_tokens":0}}
  (b) last-chunk usage = {"completion_tokens":24,"prompt_tokens":8,"total_tokens":32,"prompt_tokens_details":{"cached_tokens":0}}
  (b) EQUAL=true
```

**(c) mit Prompt-Cache** — gleiche Tokens und Ids, Werte minimal verschieden (warme KV-Wiederverwendung
ändert die Summationsreihenfolge), und im dritten Eintrag kippt der dritte `top_logprobs`-Platz
(`" #"` gegen `" return"`):

```
blocking [0] {"id":13,"token":"\n","bytes":[10],"logprob":-0.1300244927406311, …}
streamed [0] {"id":13,"token":"\n","bytes":[10],"logprob":-0.13420109450817108, …}
blocking [2] {"id":565,"token":" if", …,"logprob":-0.36814239621162415,"top_logprobs":[… ," #" …]}
streamed [2] {"id":565,"token":" if", …,"logprob":-0.3986988961696625, "top_logprobs":[… ," return" …]}
```

**(c) Gegenprobe mit `cache_prompt:false`** — die ersten drei Einträge **zeichengleich**, in beiden
Prompt-Formen:

```
=== control text prompt
  [0] identisch=True   [1] identisch=True   [2] identisch=True
  alle drei identisch = True
=== control token-id prompt
  [0] identisch=True   [1] identisch=True   [2] identisch=True
  alle drei identisch = True
```

**(d) Prompt als Token-Ids.** `LlamaModel.encode("def fibonacci(n):")` ergibt
`[822,18755,265,21566,29898,29876,1125]` (n=7). Derselbe Text wie in (a), `EQUAL=true`, `usage`
`prompt_tokens: 7`; mit `cache_prompt:false` auch `usage` und Logprobs gleich. Wichtig: die
gestreamte Anfrage mit Token-Id-Prompt wird **angenommen** — genau das, was F186 reparierte.

**(e) zwei Prompts** — beide Wege `400`, die Meldung nennt die Zahl:

```
  (e) blocking status=400 body={"error":{"message":"a request takes one prompt, this one has 2 prompts; send them as separate requests","type":"invalid_request_error","code":null}}
  (e) streamed status=400 body={"error":{"message":"a request takes one prompt, this one has 2 prompts; send them as separate requests","type":"invalid_request_error","code":null}}
  (e) blocking mentions 2 prompts=true
  (e) streamed mentions 2 prompts=true
```

Nebenbeobachtungen: `contentType` ist blockierend `application/json; charset=utf-8` und gestreamt
`text/event-stream; charset=utf-8`; alle 25 Chunks eines Streams tragen **eine** Id
(`chatcmpl-…`, 32 Zeichen), 24 davon einen `logprobs.content`-Eintrag, der letzte das `usage` und
ein textuelles `finish_reason`; `model` ist in beiden Wegen der Modellpfad, weil die Probe kein
`modelId` setzt, das vom Standard abweicht (dazu M5).

**Folgerung:** Die Hypothese ist bestätigt. (a), (d) und (e) ohne Vorbehalt; (b) und (c) ebenfalls,
sobald der Prompt-Cache ausgeschaltet ist — die beobachtete Abweichung ist allein Cache-Wirkung und
kein Unterschied zwischen den beiden Wegen. Der gestreamte Weg liefert damit denselben Text,
dieselben Zahlen und dieselben Wahrscheinlichkeiten wie der blockierende und nimmt den
Token-Id-Prompt an. Wer das Paar künftig vergleicht, muss `cache_prompt:false` setzen oder beide
Seiten gleich warm fahren, sonst vergleicht er den Cache mit.

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

**Ergebnis:** Alle drei Anfragen `200`, blockierend `application/json; charset=utf-8`, gestreamt
`text/event-stream; charset=utf-8`. Wörtlich:

**Mit `stop_sequences: ["5"]`** — der Text endet vor der 5:

```
BLOCKING body={"id":"msg_6dbc1d7e","type":"message","role":"assistant","model":"m",
 "content":[{"type":"text","text":"1, 2, 3, 4, "}],
 "usage":{"input_tokens":42,"output_tokens":13,"cache_read_input_tokens":0},
 "stop_reason":"end_turn","stop_sequence":"5"}
STREAMED message_delta={"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":"5"},
 "usage":{"input_tokens":1,"output_tokens":13,"cache_read_input_tokens":41}}
STREAMED text=[1, 2, 3, 4, ]
```

**Ohne `stop_sequences`** — die Zählung läuft durch, `stop_sequence` ist `null`:

```
BLOCKING text=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
BLOCKING stop_reason="end_turn" stop_sequence=null
STREAMED message_delta={"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},
 "usage":{"input_tokens":1,"output_tokens":30,"cache_read_input_tokens":41}}
```

**Mit `max_tokens: 3`**:

```
BLOCKING text=[1, ]
BLOCKING stop_reason="max_tokens" stop_sequence=null
STREAMED message_delta={"type":"message_delta","delta":{"stop_reason":"max_tokens","stop_sequence":null},
 "usage":{"input_tokens":1,"output_tokens":3,"cache_read_input_tokens":41}}
```

`__verbose` kommt in **keiner** der sechs Antworten und in keinem Ereignis vor (je `has
__verbose=false` über den vollständigen Körper bzw. den vollständigen SSE-Strom geprüft, nicht nur
über einzelne Felder).

**Folgerung:** Die Hypothese ist in allen drei Fällen exakt bestätigt — `end_turn` mit
`stop_sequence: "5"` und ein Text, der vor dem Stoppwort endet; `null` ohne Stoppwort; `max_tokens`
mit `null` am Budgetende; blockierend und gestreamt tragen dasselbe Paar. F183 ist mit einem echten
Stoppwort belegt.

Eine Beobachtung ohne Befundcharakter: das `usage` der gestreamten Anfrage zählt `input_tokens: 1`
bei `cache_read_input_tokens: 41`, weil die identische Anfrage unmittelbar nach der blockierenden
läuft und den warmen Prompt-Cache trifft — dieselbe Cache-Wirkung wie in M2(b), nicht ein
Unterschied der beiden Wege.

---

## M4 F182: ein gestreamter Fehler vor dem ersten Chunk mit dem echten Modell

**Ziel.** Ein Prompt über der Kontextgröße ist bei upstream `400` (`exceed_context_size_error`), gestreamt
wie nicht gestreamt. Mit `OpenAiCompatServer` über `DRAFT_MODEL_PATH` und `setCtxSize(256)` eine
Chat-Anfrage mit ~2000 Wörtern Inhalt an alle fünf Streaming-Routen (`/v1/chat/completions`,
`/v1/completions`, `/api/chat`, `/v1/messages`, `/v1/responses`, jeweils mit `"stream": true`).
Aufzeichnen: HTTP-Status, `Content-Type`, Körper.

**Hypothese.** Überall `400` mit JSON-Körper im Fehlerformat der Route, kein SSE.

**Ergebnis:** Prompt 2000 Wörter / 13 789 Zeichen, Modell `AMD-Llama-135m-code.Q2_K.gguf` mit
`setCtxSize(256)`. **Alle fünf Routen antworten `400` mit JSON, kein SSE** — geprüft wurde nicht nur
der Status, sondern auch, dass der Körper kein `data: ` und kein `event: ` enthält:

| Route | Status | `Content-Type` | SSE? |
|---|---|---|---|
| `/v1/chat/completions` | 400 | `application/json; charset=utf-8` | nein |
| `/v1/completions` | 400 | `application/json; charset=utf-8` | nein |
| `/api/chat` | 400 | `application/json; charset=utf-8` | nein |
| `/v1/messages` | 400 | `application/json; charset=utf-8` | nein |
| `/v1/responses` | 400 | `application/json; charset=utf-8` | nein |

Die Körper, je im Fehlerformat der Route:

```
/v1/chat/completions {"error":{"message":"request (5818 tokens) exceeds the available context size (256 tokens), try increasing it","type":"invalid_request_error","code":null}}
/v1/completions      {"error":{"message":"request (5791 tokens) exceeds the available context size (256 tokens), try increasing it","type":"invalid_request_error","code":null}}
/api/chat            {"error":"request (5818 tokens) exceeds the available context size (256 tokens), try increasing it"}
/v1/messages         {"type":"error","error":{"type":"invalid_request_error","message":"request (5818 tokens) exceeds the available context size (256 tokens), try increasing it"}}
/v1/responses        {"error":{"message":"request (5818 tokens) exceeds the available context size (256 tokens), try increasing it","type":"invalid_request_error","code":null}}
```

Gegenprobe, dieselbe Anfrage **ohne** `"stream"` an `/v1/chat/completions`: `400`,
`application/json; charset=utf-8`, Körper zeichengleich mit der gestreamten Zeile oben.

**Folgerung:** Die Hypothese ist vollständig bestätigt. Der gestreamte Weg verhält sich wie der nicht
gestreamte — Status, `Content-Type` und Körper sind gleich —, und keine Route schickt ein
SSE-Vorspiel, bevor sie den Fehler meldet. Dass `/v1/completions` 5791 statt 5818 Tokens nennt, ist
der fehlende Chat-Template-Rahmen, nicht eine Abweichung. F182 ist mit dem echten Modell auf allen
fünf Routen belegt.

---

## M5 F185: `id` und `model` über beide Server

**Ziel.** (a) `OpenAiCompatServer` mit `--model-id my-model` (Kommandozeile über `ServerLauncher
--jllama-openai-compat`): `/v1/models`, eine blockierende Chat-Antwort und ein Stream; jedes `id` beginnt
mit `chatcmpl-`, alle Chunks eines Streams tragen dieselbe, `model` ist überall `my-model`. (b) Dasselbe
ohne `--model-id` (erwartet: der Dateiname). (c) `NativeServer` mit `--alias my-model` als Gegenprobe gegen
upstreams eigenen Server im selben Prozess.

**Ergebnis:** Je ein echtes Kind-JVM über `ServerLauncher`, Modell `AMD-Llama-135m-code.Q2_K.gguf`,
`-c 512 -ngl 0`; der Port wird aus der Startmeldung des Kindes gelesen und mit `/health` bestätigt.

**(a) `--jllama-openai-compat … --model-id my-model`** — wie erwartet:

```
GET /v1/models  {"object":"list","data":[{"id":"my-model","object":"model","created":1792447453,"owned_by":"llama.cpp","ftype":"Q2_K - Medium"}]}
blocking id=chatcmpl-14XgXnJnNaDyD6DdDBXV6g5uuEY4N9Q6  startsWith chatcmpl-=true
blocking model=my-model  == requested=true
streamed chunks=8 ids=[chatcmpl-1CHN9eEjZyRnHKJdRENc6eBH7pRXQRn7] models=[my-model]
streamed one id=true  id startsWith chatcmpl-=true
```

Also: jede `id` beginnt mit `chatcmpl-`, alle acht Chunks eines Streams tragen **eine** Id, und
`model` ist überall `my-model`.

**(c) `NativeServer --alias my-model`** als Gegenprobe gegen upstreams eigenen Server — deckungsgleich:

```
GET /v1/models  … "data":[{"id":"my-model","aliases":["my-model"], … }]
blocking id=chatcmpl-fqJZm8uzg7akXK4lPIOnH9br0hNIYpqY  startsWith chatcmpl-=true
blocking model=my-model  == requested=true
streamed chunks=8 ids=[chatcmpl-EWLKYE2i6pFSMWGhlXs5JukSRzbxDHjk] models=[my-model]
```

**(b) ohne `--model-id` — hier weicht der Lauf ab, und das ist ein Befund.**

**Befund B2 (Fundbuch: F189): ohne `--model-id` benennen `/v1/models` und die Antworten dasselbe Modell verschieden.**

```
GET /v1/models  {"object":"list","data":[{"id":"AMD-Llama-135m-code.Q2_K.gguf", … }]}
blocking model=X:\Privat\OpenSource\github\java-llama.cpp\models\AMD-Llama-135m-code.Q2_K.gguf
streamed models=[X:\Privat\OpenSource\github\java-llama.cpp\models\AMD-Llama-135m-code.Q2_K.gguf]
```

`/v1/models` meldet den **Dateinamen** — genau was die Usage-Zeile verspricht („`--model-id <name>`
Model id reported by /v1/models (default: file name)") und was
`OpenAiServerCli.ParsedArgs.getModelId()` aus `Paths.get(modelPath).getFileName()` ableitet. Die
Antworten melden dagegen den **vollen absoluten Pfad**, weil das `model`-Feld der Antwort nicht aus
der Konfiguration kommt, sondern aus llama.cpps `--alias`, und das wird nur unter einer Bedingung
gesetzt (`OpenAiServerCli.toModelParameters()`):

```java
if (modelId != null) {
    // the answers name the model as llama.cpp knows it; this keeps them equal to /v1/models
    params.setAlias(modelId);
}
```

Der Kommentar nennt die Absicht — Antworten und `/v1/models` gleich halten —, aber der Zweig läuft
nur, wenn `--model-id` wirklich übergeben wurde. Ohne das Flag bleibt der Alias leer und upstream
setzt den Modellpfad ein. (c) zeigt, dass `--alias` die eine wirksame Stelle ist: damit stimmen beide
Flächen. Die `id`-Hälfte von F185 ist davon **nicht** betroffen — `chatcmpl-` und eine Id pro Stream
gelten in allen drei Fällen.

Wirkung: ein Client, der `/v1/models` liest und das `model`-Feld der Antwort dagegen hält — oder der
den aus `/v1/models` gelesenen Namen in die nächste Anfrage zurückschreibt —, sieht ohne
`--model-id` zwei verschiedene Namen. Zusätzlich steht dann ein absoluter lokaler Pfad in jeder
Antwort, auch in einer über das Netz.

Dieselbe Trennung erklärt eine Beobachtung aus M2: die dortige Probe setzt `modelId("m")` direkt über
`OpenAiServerConfig` (nicht über die CLI), und die Antworten tragen trotzdem den Modellpfad — weil
nur die CLI `setAlias` aufruft. Über die eingebettete API ist das Feld also gar nicht einstellbar.

**Folgerung:** (a) und (c) bestätigen F185 vollständig. (b) wich ab: der Vorgabewert griff nur für
`/v1/models`, nicht für das `model`-Feld der Antworten, sodass die im Code ausgesprochene Zusage
„keeps them equal to /v1/models" ohne das Flag nicht galt.

### B2 (F189) ist behoben (auf Weisung, nach der Messung)

**Warum der Pfad überhaupt erschien, nachgelesen statt vermutet.** Upstreams Kette für den Modellnamen
(`tools/server/server-context.cpp`) ist Alias → `common_params_model::get_name()` → Dateiname, und
`get_name()` lautet:

```cpp
std::string get_name() const {
    if (!hf_repo.empty())     { return hf_repo; }
    if (!docker_repo.empty()) { return docker_repo; }
    return path;
}
```

Für eine lokale Datei gibt Schritt 2 also den **Pfad** zurück, womit Schritt 3 — upstreams eigener
Kommentar „fallback: derive model name from file name" — unerreichbar ist. Ohne Alias nennt der Server
darum den Pfad, auch der von upstream selbst.

**Der Fix** ist eine Zeile in `OpenAiServerCli.toModelParameters()`: der Alias wird **immer** gesetzt,
aus `getModelId()` statt nur aus einem übergebenen `modelId`. Damit trägt der Vorgabewert dieselbe
Zusage wie das Flag, und der eine Name, den der Server bewirbt, ist der, den llama.cpp benutzt.

**End-to-end belegt** (Wegwerf-Probe, danach entfernt), derselbe Fall wie (b), ohne `--model-id`:

```
  expected id (file name)=AMD-Llama-135m-code.Q2_K.gguf
  GET /v1/models id = AMD-Llama-135m-code.Q2_K.gguf
  blocking model    = AMD-Llama-135m-code.Q2_K.gguf
  blocking id       = chatcmpl-rkhNFoYimoN16qjlHGQNTFGFnWhvXRxw
  streamed models   = [AMD-Llama-135m-code.Q2_K.gguf]
  AGREE blocking == /v1/models : true
  AGREE streamed == /v1/models : true
  id is chatcmpl- and one per stream : true
  no absolute path in the answer     : true
```

Vorher nannten dieselben drei Stellen den vollen absoluten Pfad; der lokale Pfad steht jetzt auch
nicht mehr in jeder Antwort über das Netz.

**Wächter:** `OpenAiServerCliTest.theAdvertisedModelIdBecomesTheModelsAliasWithAndWithoutTheFlag` —
vorher hieß der Test `anExplicitModelIdBecomesTheModelsAlias` und schrieb das Gegenteil fest (ohne
Flag *kein* `--alias`); er prüft nun beide Fälle und dass Alias und `getModelId()` derselbe String
sind.

**Was bewusst offen bleibt: der eingebettete Weg.** `OpenAiCompatServer(model, config)` bekommt ein
schon geladenes Modell, kann also kein `--alias` mehr setzen — die Beobachtung aus M2 (dort steht der
Pfad im `model`-Feld, obwohl die Probe `modelId("m")` setzt) gilt weiter. Das ist jetzt im Javadoc von
`OpenAiServerConfig.getModelId()` gesagt, mit dem Hinweis, dass ein einbettender Aufrufer sein Modell
mit `ModelParameters.setAlias(…)` laden muss. Das `model`-Feld der Antwort nachträglich umzuschreiben
wäre die Alternative; sie würde über die Engine etwas anderes behaupten als diese getan hat und würde
von `NativeServer` im Attach-Modus ohnehin nicht nachvollzogen — daher nicht getan, sondern benannt.

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

**Ergebnis:** Zuerst ein Befund, der den Ablauf änderte, dann die Messung selbst.

**Befund B1 (Fundbuch: F187): `ModelParameters.addLoraScaledAdapter` ist gegen den Pin unbrauchbar.** Der erste
Versuch, Referenz B über `addLoraScaledAdapter(ADAPTER, 1.0f)` zu laden, scheiterte:

```
net.ladenthin.llama.exception.LlamaException: Failed to parse model parameters
	at net.ladenthin.llama.LlamaModel.loadModel(Native Method)
```

Die Methode rendert `putString(ModelOption.LORA_SCALED, fname + "," + scale)`, also
`--lora-scaled <FNAME>,<SCALE>`. Upstream liest den Wert aber als `FNAME:SCALE,...` —
`common/arg.cpp` am Pin:

```
{"--lora-scaled"}, "FNAME:SCALE,...",
"path to LoRA adapter with user defined scaling (format: FNAME:SCALE,...)\nnote: use comma-separated values",
[](common_params & params, const std::string & value) {
    for (const auto & item : parse_csv_row(value)) {
        auto parts = string_split<std::string>(item, ':');
        if (parts.size() != 2) { throw std::invalid_argument("lora-scaled format: FNAME:SCALE"); }
```

Das Komma ist seit **b7448** der Listentrenner, nicht der Feldtrenner: `487674fbb` („common: fix
--override-kv to support comma-separated values", upstream #18056, 2025-12-17), später erweitert um
`07fbe19f1` (b7648, „arg: use CSV escape style for multiple-value args", #18643). Der Pin ist
b11639, die Methode ist also seit rund 4200 Builds tot. **`addControlVectorScaled` hat dieselbe
Form** (`fname + "," + scale`) und `--control-vector-scaled` dasselbe upstream-Format, ist also
gleichermaßen betroffen. `setOverrideKv(String)` übergibt den Wert unverändert, liegt damit beim
Aufrufer.

Abgedeckt ist das nur von `ModelParametersTest#testAddLoraScaledAdapter`, das genau das falsche
Format festschreibt:

```java
ModelParameters p = new ModelParameters().addLoraScaledAdapter("adapter.bin", 0.5f);
assertThat(p.parameters.get("--lora-scaled"), is("adapter.bin,0.5"));
```

Das ist die Klasse, die CLAUDE.md für die Wire-Name-Registries beschreibt — ein Test, der die
String-Abbildung prüft und darum grün bleibt, während das Flag tot ist. Die Registry-Prüfung
`test_model_flags.cpp` sieht es nicht: sie hält den **Namen** gegen die Optionstabelle des Servers,
nicht das **Werteformat**.

Alle vier Formen gegen den echten Parser (je ein Kind-JVM über `ServerLauncher` → `NativeServer`,
der llama-server-argv unverändert weitergibt):

```
1. --lora-scaled <abs>,1.0   (was addLoraScaledAdapter heute sendet)
   -> error while handling argument "--lora-scaled": invalid stof argument      Server startet nicht
2. --lora-scaled <abs>:1.0   (upstream-Format, absoluter Windows-Pfad)
   -> error while handling argument "--lora-scaled": lora-scaled format: FNAME:SCALE   Server startet nicht
3. --lora-scaled <rel>:1.0   (upstream-Format, relativer Pfad)
   -> 0.00.202.912 I srv  llama_server: listening on http://127.0.0.1:64459     angenommen
4. --lora <abs>              (Gegenprobe, Skala 1.0 per Vorgabe)
   -> 0.00.202.540 I srv  llama_server: listening on http://127.0.0.1:61766     angenommen
```

Form 2 ist bemerkenswert: **auch das korrekte Doppelpunkt-Format trägt auf Windows keinen absoluten
Pfad.** `string_split` trennt an *jedem* Doppelpunkt (`common/common.h`), also ergibt
`X:\…\a.gguf:1.0` drei Teile und nicht zwei. Eine Korrektur am Projekt von `,` auf `:` würde auf
Linux greifen, auf Windows mit Laufwerksbuchstaben weiter scheitern — das ist eine
upstream-Einschränkung, keine Projektentscheidung. (Nebenbei: der Parse-Fehler endet den Prozess mit
**Exit 0** nach der Usage-Ausgabe.)

Gemessen wurde darum mit `addLoraAdapter(ADAPTER)` (`--lora`, Skala 1.0 per Vorgabe) — genau die
Skala, die M6 verlangt.

**Die Messung.** Modell `stories15M_MOE-F16.gguf`, Adapter `moe_shakespeare15M.gguf`, Prompt „Look in
thy glass", 32 Tokens, `temperature 0, top_k 1, seed 42, cache_prompt true, id_slot 0`,
`setParallel(1)`, Kontext 512. Der Server meldet den Adapter als
`[{"id":0,"path":"…moe_shakespeare15M.gguf","scale":1.0,…}]`.

| Schritt | `cacheRamMib(0)` | Voreinstellung |
|---|---|---|
| A ohne Adapter | `was a little girl. She was only three years old …` | gleich |
| B Adapter 1.0, kalt | `in love of the eye:\nThat's when when the eye see thy on the sun's shining bright. …` | gleich |
| B′ Adapter 1.0, warm | = B | = B |
| C nach `setLoraAdapter(0, 0f)` | **= A** | **= A** |
| D nach `setLoraAdapter(0, 1f)` | **= B** | **= B** |
| E Anfrage mit eigenem `lora` Skala 0 | **= A** | **= A** |
| F nächste Anfrage, Server-Adapter 1.0 | **= B** | **= B** |

Wörtlich aus dem Durchgang mit der Voreinstellung — dem, den der Auftrag als offenen Rest markiert:

```
  C  (setLoraAdapter(0,0f), warm slot)   = [ was a little girl. She was only three years old …]
     C == A (expected) : true
     C == B (defect)   : false
  D  (setLoraAdapter(0,1f), warm slot)   = [ in love of the eye: …]
     D == B (expected) : true
     D == A (defect)   : false
  E  (request lora scale 0)              = [ was a little girl. …]
     E == A (expected) : true
  F  (next request, server adapters 1.0) = [ in love of the eye: …]
     F == B (expected) : true
     F == A (defect)   : false
```

**Folgerung:** Die Messung bestätigt Patch 0021 mit dem echten Adapterpaar, und sie schließt den
offenen Rest: **der Host-Speicher-Cache weicht nicht ab.** Beide Durchgänge sind Schritt für Schritt
gleich — ein Adapterwechsel am warmen Slot wird nie aus dem alten Cache beantwortet, in keiner der
beiden Cache-Einstellungen, auch nicht nach einer Anfrage mit eigenen Adaptern. Der Vorbehalt im
Javadoc von `setLoraAdapters` lässt sich mit dieser Messung nicht nachstellen.

Offen blieb damit nicht Patch 0021, sondern Befund **B1**: der skalierte Adapterpfad über
`ModelParameters` lud am Pin überhaupt kein Modell, und der einzige Test darüber schrieb das falsche
Format fest. Die Messung musste deshalb auf `--lora` ausweichen.

### B1 (F187) und B4 (F188) sind behoben (auf Weisung, nach der Messung)

Beim Fixen kam ein weiterer Fehler derselben Familie mit heraus, **B4** (Fundbuch: F188): `putString` ist
`map.put(option, value)` auf einer `LinkedHashMap`, also **überschreibt** ein zweiter
`addLoraAdapter` / `addLoraScaledAdapter` / `addControlVector` / `addControlVectorScaled` den ersten —
während alle vier im Javadoc „can be repeated to use multiple adapters" versprechen. Upstream liest
genau diese Optionen seit #18056 / #18643 als **eine** komma-separierte Liste, Akkumulation ist also
das, was sie zusammensetzt.

Der Fix, in `llama/src/main/java/net/ladenthin/llama/parameters/`:

- **`CliParameters.appendCsvValue(option, value)`** neu: hängt einen Wert an, statt zu ersetzen, und
  schreibt ihn als CSV-Feld — in Anführungszeichen mit verdoppelten inneren Anführungszeichen, sobald
  er ein Komma oder ein Anführungszeichen trägt, weil upstreams `parse_csv_row` ein öffnendes
  Anführungszeichen nur am Feldanfang erkennt und ein Pfad mit Komma sonst gar nicht übergeben werden
  kann.
- **Die vier Methoden** nutzen ihn; die beiden skalierten rendern `FNAME:SCALE` statt `FNAME,SCALE`.
- **Ein Pfad mit Doppelpunkt wird laut abgelehnt** (`IllegalArgumentException`), statt den Ladevorgang
  später mit „Failed to parse model parameters" scheitern zu lassen. Die Meldung nennt die Ursache und
  die zwei Auswege. Das ist die Grenze aus Form 2 oben: sie ist upstreams, nicht unsere, und lässt
  sich hier nicht wegkorrigieren — ein `,`→`:`-Tausch allein hätte auf Windows weiter nicht geladen.

**End-to-end belegt** mit demselben Adapterpaar (Wegwerf-Probe, danach entfernt):

```
### B1 absolute path refused: --lora-scaled reads its value as FNAME:SCALE and splits on every colon,
    so a path containing one cannot be expressed: X:\…\moe_shakespeare15M.gguf
    -- pass a relative path, or load it unscaled and set the scale after the load
  rendered argv for the relative path: [, --fit, on, --lora-scaled, ..\models\moe_shakespeare15M.gguf:0.5,second.gguf:0.25]
  A  (no adapter)                  = [ was a little girl. She was only three years old …]
  B  (--lora, scale 1.0)           = [ in love of the eye: …]
  C  (--lora-scaled <rel>:1.0)     = [ in love of the eye: …]
     adapters: [{"id":0,"path":"..\\models\\moe_shakespeare15M.gguf","scale":1.0,…}]
     C == B (expected) : true
  D  (--lora-scaled <rel>:0.0)     = [ was a little girl. …]
     adapters: [{"id":0,"path":"..\\models\\moe_shakespeare15M.gguf","scale":0.0,…}]
     D == A (expected, scale carried) : true
     D == B (defect, scale ignored)   : false
```

Also: die Option **lädt überhaupt erst** (vorher `LlamaException: Failed to parse model parameters`),
zwei skalierte Adapter sammeln sich zu einem Wert, und die Skala wirkt — `:1.0` ergibt Referenz B,
`:0.0` ergibt Referenz A, und der Server meldet die Skala zurück. Damit ist auch das nachgeholt, was
die Messung nicht konnte: eine **andere** Skala als 1.0 über `ModelParameters`.

**Wächter**, in `ModelParametersTest`: das neue `FNAME:SCALE`-Format (die beiden Tests, die das alte
festschrieben, sind umgestellt), die Akkumulation aller vier Optionen, das CSV-Quoting eines Pfads mit
Komma und Anführungszeichen, und die Ablehnung eines Doppelpunkt-Pfads samt der Meldung — mit einem
relativen Pfad als Gegenprobe im selben Test. 97 Tests der Klasse grün.

---

## M7 F181: Fehlermeldungen mit einem 4-Byte-Zeichen auf Windows

**Ziel.** `LlamaTrainerTest` läuft in CI auf Linux; der Windows-Weg (UTF-16-Dateinamen, `cmd`-Codepage)
ist ungeprüft. Auf Windows: `mvn -B -f llama/pom.xml test -Dtest=LlamaTrainerTest`, dann eine Wegwerf-Probe,
die `new LlamaModel(new ModelParameters().setModel("C:/tmp/fehlt-\uD83D\uDE00.gguf"))` und
`NativeServer` mit demselben Pfad startet und die Meldung codepunktweise ausgibt.

**Hypothese.** Beide Tests grün (nicht übersprungen); die Probe zeigt U+1F600 als einen Codepunkt, nichts
abgeschnitten.

**Ergebnis:** Beide Teile grün.

**`LlamaTrainerTest` auf Windows** — die beiden Methoden liefen, nicht übersprungen (aus M1):

```
[INFO] Tests run: 2, Failures: 0, Errors: 0, Skipped: 0 -- in net.ladenthin.llama.LlamaTrainerTest
testcase name="aFourByteCharacterInAPathComesBackIntactInTheErrorMessage"
testcase name="aFourByteCharacterInTheTrainingTextReachesTheTrainer"
```

**Die Probe.** Umgebung des Test-JVM:

```
os=Windows 11 10.0 | file.encoding=UTF-8 | sun.jnu.encoding=Cp1252 | native.encoding=Cp1252 | java=21.0.12.1+1-LTS
path asked for=[C:\Users\…\junit-4414633441307162606\fehlt-😀.gguf] codePointCount=72
Paths.get round trip equal=true
```

`sun.jnu.encoding` ist also **Cp1252** und kann U+1F600 gar nicht darstellen — der interessante Fall.
Die Meldung des Trainers, codepunktweise:

```
### M7 LlamaTrainer.finetune -> net.ladenthin.llama.exception.LlamaException
  message=[failed to load model for training: C:\…\junit-4414633441307162606\fehlt-😀.gguf]
  length(chars)=108 codePointCount=107
  codepoints=… U+0066 U+0065 U+0068 U+006C U+0074 U+002D U+1F600 U+002E U+0067 U+0067 U+0075 U+0066
  contains U+1F600 as ONE codepoint=true
  mangled as U+00F0 U+009F U+0098 U+0080=false
  truncated (ends in ".g" instead of ".gguf")=false
  replacement char U+FFFD present=false
  '?' mangling present=false
  EQUALS expected=true
```

Die Codepunktfolge endet also auf `f e h l t - U+1F600 . g g u f`: ein Codepunkt für das Emoji,
`.gguf` vollständig, und die Meldung ist zeichengleich mit dem erwarteten String. Die
Modified-UTF-8-Verstümmelung, die F181 behob (U+00F0 U+009F U+0098 U+0080 und ein nach `.g`
abgeschnittener Rest), tritt nicht auf. Dasselbe mit dem 4-Byte-Zeichen im **Trainingstext** statt im
Pfad: Meldung unverstümmelt, Pfad korrekt.

Zwei Nebenbeobachtungen, beide **kein** Befund:

- `new LlamaModel(…)` mit demselben Pfad wirft `LlamaException: could not load model from given file
  path` — die Meldung trägt den Pfad **gar nicht**, dieser Eintrittspunkt kann F181 also nicht
  zeigen. Der Trainer ist die Stelle, an der der Pfad in der Meldung steht.
- Die **nativen Logzeilen** auf der Konsole zeigen `fehlt-?.gguf`
  (`gguf_init_from_file: failed to open GGUF file 'C:\…\fehlt-?.gguf'`). Das ist die Codepage der
  `cmd`-Konsole beim Schreiben von stderr, nicht Datenverlust: die Java-sichtbare Meldung derselben
  Anfrage trägt U+1F600 unverändert (oben). Wer die Konsole als Beleg nimmt, kommt zum falschen
  Schluss.
- `new NativeServer("--model", <emoji-pfad>, "--port", "0").start()` kehrt ohne Ausnahme zurück. Das
  ist der dokumentierte Vertrag — „Starts the native HTTP server … and returns immediately. The
  server binds and begins serving `GET /health` before the model finishes loading; poll `/health` for
  readiness" —, also erwartet und kein Befund.

**Folgerung:** Die Hypothese ist bestätigt. Beide Trainer-Tests laufen auf Windows grün und sind
nicht übersprungen, und die Probe zeigt U+1F600 als einen Codepunkt bei intaktem Rest, obwohl
`sun.jnu.encoding=Cp1252` das Zeichen nicht darstellen kann. F181 trägt auf dem Windows-Weg.

---

## M8 F182 im Agenten: Atmosphere über das echte Modell

**Ziel.** Seit F182 meldet der Agent einen Fehler vor dem ersten Chunk als Fehler des Zuges statt still zu
enden. Lauf: `mvn -B -f llama-atmosphere-agent/pom.xml -Dllama.version=6.0.0-SNAPSHOT -Dllama.natives=none
verify` gegen den installierten Kern (`mvn -pl llama,llama-platform -am install -DskipTests`), dann
`AtmosphereToolLoopIntegrationTest` mit dem Tool-Modell (`-Dnet.ladenthin.llama.lib.path=<natives-Verzeichnis>`).
Dazu ein Zug im Agenten mit einem Prompt über der Kontextgröße (`--ctx-size 256`): was zeigt die Konsole?

**Ergebnis:** In drei Teilen — der modellfreie Lauf (mit einem Befund), der Tool-Loop mit dem echten
Modell, und der Zug über der Kontextgröße.

**(1) Modellfreier Lauf.** `mvn -f llama-atmosphere-agent/pom.xml -Dllama.version=6.0.0-SNAPSHOT
-Dllama.natives=none verify` gegen den installierten Kern (`mvn -pl llama,llama-platform -am install
-DskipTests`, `BUILD SUCCESS`):

```
[ERROR] Tests run: 276, Failures: 1, Errors: 1, Skipped: 60
[INFO] BUILD FAILURE
```

Die 60 übersprungenen sind erwartet und dokumentiert (`ScreenUseCasesTest` 57 + `ReflowResizeTest` 3,
die sich gegen das veröffentlichte JLine selbst abschalten, siehe `llama-atmosphere-agent/CLAUDE.md`).
Der `Errors: 1` ist mein Aufruf: `AtmosphereToolLoopIntegrationTest` findet lokal sein GGUF (anders als
in CIs modellfreiem Job) und läuft dann in `UnsatisfiedLinkError: No native library found`, weil
`-Dllama.natives=none` keine Natives auf den Klassenpfad legt und ich `lib.path` erst in Teil (2)
gesetzt habe. Kein Projektfehler, aber eine Folge, die lokal jeden trifft, der den modellfreien Lauf
mit vorhandenen Modellen fährt.

**Der `Failures: 1` ist ein Befund.**

**Befund B3 (Fundbuch: F190): `AgentSessionTest.aLoopRunsItsStepsThroughTheFrontEndUntilTheMarker` ist auf Windows
deterministisch rot.** Drei Läufe, drei Mal dieselbe Zeile:

```
Lauf 1 : [ERROR] Tests run: 13, Failures: 1, Errors: 0, Skipped: 0
Lauf 2 : [ERROR] Tests run: 13, Failures: 1, Errors: 0, Skipped: 0
Lauf 3 : [ERROR] Tests run: 13, Failures: 1, Errors: 0, Skipped: 0
```

```
java.lang.AssertionError:
Expected: a string containing "loop: done after 1 steps"
     but: was "loop: write a poem
loop: state in C:\Users\…\AGENT-LOOP.md, max 20 steps, budget 120 min
── loop step 1/20 ──
── loop step 2/20 ──
── loop step 3/20 ──
loop: no progress for 3 steps (nothing written, no tools used)"
	at …AgentSessionTest.aLoopRunsItsStepsThroughTheFrontEndUntilTheMarker(AgentSessionTest.java:341)
```

und unmittelbar davor im Log:

```
WARN org.atmosphere.ai.llm.OpenAiCompatibleClient - LLM connection error, retrying in 500ms (attempt 1/3): HTTP/1.1 header parser received no bytes
```

Ursache, mit einer Probe belegt statt erschlossen. Der Test lässt das Skript-Backend
`"working" + System.lineSeparator() + TaskLoop.SENTINEL` senden, und `ScriptedBackend.quote`
(`llama-atmosphere-agent/src/test/.../ScriptedBackend.java`) escapt Anführungszeichen, Backslash und
`\n` — **nicht `\r`**: der `switch` hat Zweige für `'"'`, `'\\'` und `'\n'` und schreibt alles andere
unverändert durch.

Auf Windows ist `System.lineSeparator()` `\r\n`, der Chunk trägt also ein rohes CR mitten im
JSON-String. Die Probe zeigt beides nebeneinander:

```
os=Windows 11  System.lineSeparator() = CRLF
### as the test builds it: "working" + System.lineSeparator() + SENTINEL
  text   = working<0D><0A><<TASK_COMPLETE>>
  chunk  = {… "delta":{"content":"working<0D>\n<<TASK_COMPLETE>>"} …}
  raw CR in chunk  = 113
  parses as JSON   = NO: JsonParseException: Illegal unquoted character ((CTRL-CHAR, code 13)): has to be escaped using backslash to be included in string value
  TaskLoop.isComplete(text) = true
### with LF only (what Linux gets)
  text   = working<0A><<TASK_COMPLETE>>
  chunk  = {… "delta":{"content":"working\n<<TASK_COMPLETE>>"} …}
  raw CR in chunk  = -1
  parses as JSON   = yes
  TaskLoop.isComplete(text) = true
```

Der Chunk ist also auf Windows **kein gültiges JSON** (RFC 8259 verbietet unescapte Steuerzeichen
U+0000–U+001F), die SSE-Antwort bricht ab, jeder Loop-Schritt bekommt nichts, und der
Stillstandswächter beendet den Loop nach drei Schritten. **Die Sentinel-Erkennung ist nicht
schuld** — `TaskLoop.isComplete` ist für beide Zeilenenden `true`.

Es ist Testcode, kann also nichts an Nutzer ausliefern; die Wirkung ist, dass das Agentenmodul auf
Windows nicht grün baut. Gesehen wurde es nie, weil die drei Agenten-Jobs nur auf Linux laufen.

**B3 (F190) ist auf Weisung in diesem Lauf behoben** — eine Ausnahme von der Regel „nur messen", ausdrücklich
beauftragt. Zwei Dinge dabei, die beim Messen auffielen:

**Die Lücke war viel größer als das fehlende `\r`.** Beide `quote`-Fassungen in einem
Einzeldatei-Programm gegen Jackson, auf dieser Maschine:

```
System.lineSeparator() = CRLF
VORHER   control chars that are not JSON: 31/32   value altered: 0
         namely: U+0000 U+0001 … U+000D … U+001F
         System.lineSeparator() round trip: FAILS: Illegal unquoted character ((CTRL-CHAR, code 13))
NACHHER  control chars that are not JSON:  0/32   value altered: 0
         System.lineSeparator() round trip: ok
```

Von 32 Steuerzeichen erzeugten **31** ungültiges JSON; behandelt war allein `\n`. Der Wert wird in
keiner Fassung verändert, es ging rein um Gültigkeit.

**Der Fix** setzt die vier benannten Escapes `\r \t \b \f` dazu und — das Tragende — einen
`default`-Zweig, der alles unter `0x20` als `\uXXXX` schreibt, damit ein Steuerzeichen, das niemand
aufgelistet hat, die Lücke nicht wieder aufreißen kann. Der Test selbst bleibt unangetastet:
`System.lineSeparator()` dort ist richtig, weil genau das CRLF überhaupt erst ausübt.

**Wächter: `ScriptedBackendTest`** (neu, 3 Fälle) — jeder Chunk eines Text-Zuges muss als JSON
parsen (mit `System.lineSeparator()`, CRLF, LF, einzelnem CR); **jedes** der 32 Steuerzeichen
einzeln, also der `default`-Zweig und nicht die vier benannten Fälle; und der Text muss unverändert
aus dem Chunk zurückkommen. Ohne so einen Wächter ist der Fehler unsichtbar, weil ein kaputter Chunk
als abgebrochener Stream ankommt und die rote Zusicherung mehrere Schichten entfernt über die
*Antwort* spricht.

**Rot/grün im echten Build belegt, nicht nur in der Isolation.** Mit der alten `quote`-Fassung kurz
zurückgesetzt:

```
[ERROR] Tests run: 13, Failures: 1, Errors: 0, Skipped: 0 -- in …AgentSessionTest
[ERROR] Tests run:  3, Failures: 2, Errors: 1, Skipped: 0 -- in …ScriptedBackendTest
[ERROR]   ScriptedBackendTest.aTextTurnCarryingTheLineSeparatorIsJsonOnEveryPlatform:44 System.lineSeparator(): chunk is not JSON: …
[ERROR]   ScriptedBackendTest.everyControlCharacterSurvivesAsJson:60 U+0000: chunk is not JSON: …
[ERROR]   ScriptedBackendTest.theTextComesBackOutOfTheChunkUnchanged:73 » JsonParse Illegal unquoted character ((CTRL-CHAR, code 13))
```

und mit dem Fix (die Datei danach byte-identisch wiederhergestellt), drei Läufe hintereinander:

```
Lauf 1/2/3: Tests run: 13, Failures: 0, Errors: 0, Skipped: 0 -- in …AgentSessionTest
Lauf 1/2/3: Tests run:  3, Failures: 0, Errors: 0, Skipped: 0 -- in …ScriptedBackendTest
```

Und das ganze Modul, diesmal mit `lib.path`, also ohne den Aufruffehler aus Teil (1):

```
[INFO] Tests run: 281, Failures: 0, Errors: 0, Skipped: 60
[INFO] BUILD SUCCESS
```

Die 60 Auslassungen sind unverändert die dokumentierten JLine-Fälle. `spotless:apply` hatte nichts zu
ändern. Geprüft wurde außerdem, ob dieselbe Lücke woanders steckt: ein `grep` über alle handgerollten
JSON-Quote-Helfer im Repository findet nur diese eine Stelle.

**(2) Der Tool-Loop mit dem echten Modell.** Dieselbe Klasse mit
`-Dnet.ladenthin.llama.lib.path=<natives>` und dem Tool-Modell:

```
[INFO] Tests run: 3, Failures: 0, Errors: 0, Skipped: 0, Time elapsed: 10.76 s -- in net.ladenthin.llama.atmosphere.AtmosphereToolLoopIntegrationTest
[INFO] BUILD SUCCESS
```

mit den Methoden `plainChatStreamsAnAnswer`, `toolCallResultIsFedBackAndAnswered` und
`multiRoundReadWriteReadLoopChangesTheFile` — nicht übersprungen.

**(3) Der Zug über der Kontextgröße.** Vier Züge über `exec:java` (einmal `--ctx-size 256` wie im
Auftrag, dazu eine brauchbare Gegenprobe, siehe unten), je `--plain --temperature 0 --ngl 0`:

Zug 1, `--ctx-size 256` mit dem 2000-Wörter-Prompt — die Konsole zeigt:

```
(context nearly full — compacting first)
(nothing to compact)
srv    send_error: task id = 0, error: request (1897 tokens) exceeds the available context size (256 tokens), try increasing it
srv          stop: cancel task, id_task = 0
ERROR org.atmosphere.ai.llm.OpenAiCompatibleClient - LLM API error (400): {"error":{"message":"request (1897 tokens) exceeds the available context size (256 tokens), try increasing it","type":"invalid_request_error","code":null}}
[error] org.atmosphere.ai.AiProviderException$TerminalProviderError: API returned 400: request (1897 tokens) exceeds the available context size (256 tokens), try increasing it
AGENT_EXIT_OVER=1
```

Zug 2, `--ctx-size 256` mit dem **kurzen** Prompt, scheitert ebenso (`request (1483 tokens) exceeds
the available context size (256 tokens)`) — **bei 256 sprengt schon der Systemprompt den Kontext**,
der im Auftrag vorgeschlagene Wert ist für den Agenten also an sich zu klein und taugt nicht als
Gegenprobe. Darum zwei weitere:

Zug 3, `--ctx-size 4096`, kurzer Prompt — der Agent antwortet normal:

```
2 + 2 equals 4.
AGENT_EXIT_CTRL=0
```

Zug 4, `--ctx-size 1024`, der 2000-Wörter-Prompt — wieder der Fehler des Zuges:

```
[error] org.atmosphere.ai.AiProviderException$TerminalProviderError: API returned 400: request (1897 tokens) exceeds the available context size (1024 tokens), try increasing it
AGENT_EXIT_OVER2=1
```

**Folgerung:** Teil (3) bestätigt F182 im Agenten: ein Fehler vor dem ersten Chunk erscheint als
`[error] … API returned 400: …` mit der vollständigen Meldung des Servers und beendet den Prozess mit
Code 1, statt still zu enden — und Zug 3 zeigt im Gegenzug, dass derselbe Agent mit arbeitsfähigem
Kontext normal antwortet, der Fehler also am Kontext hängt und nicht am Aufbau. Teil (2) bestätigt
den Tool-Loop mit dem echten Modell. Teil (1) brachte Befund **B3** — das Agentenmodul baute auf
Windows nicht grün, weil der Testharnisch 31 von 32 Steuerzeichen unescapt in seine JSON-Chunks
schrieb —, und B3 ist auf Weisung in diesem Lauf behoben und mit einem Wächter versehen; das Modul
ist damit auf Windows grün (281 Tests, 0 Fehler, 60 dokumentiert übersprungen).

Zwei Dinge, die **kein** Befund sind: das `â€"` in einer früheren Lesung von
`(context nearly full — compacting first)` war die Codepage meines Log-Lesens, nicht die Ausgabe des
Agenten (als UTF-8 gelesen steht dort ein korrektes `—`); und `--ctx-size 256` als Vorschlag im
Auftrag ist, wie oben, für den Agenten zu klein.

---

## M9 `llama-ffm` mit dem echten Modell

**Ziel.** `mvn -B -f llama-ffm/pom.xml verify -Dnet.ladenthin.llama.ffm.model=models/AMD-Llama-135m-code.Q2_K.gguf
-Dnet.ladenthin.llama.lib.path=<…/llama/src/main/natives/net/ladenthin/llama/<OS>/<ARCH>/cpu>` — neu ist
`FfmLlamaModelIntegrationTest#oaiCompletionStreamEndsWithUsage` (F186).

**Ergebnis:** Grün, nichts übersprungen. Die Toolchain greift wie vorgesehen — Maven läuft auf
Temurin 21, das Modul auf Temurin 25:

```
[INFO] --- toolchains:3.3.0:toolchain (default) @ llama-ffm ---
[INFO] Required toolchain: jdk [ version='[22,)' ]
[INFO] Found matching toolchain for type jdk: JDK[C:\Users\berna\AppData\Local\Programs\jdk-25.0.4.1+1]
[INFO] Toolchain in maven-compiler-plugin: JDK[…jdk-25.0.4.1+1]
[INFO] Toolchain in maven-surefire-plugin: JDK[…jdk-25.0.4.1+1]
```

```
[INFO] Tests run: 17, Failures: 0, Errors: 0, Skipped: 0, Time elapsed: 1.355 s -- in net.ladenthin.llama.ffm.FfmLlamaModelIntegrationTest
[INFO] Tests run:  4, Failures: 0, Errors: 0, Skipped: 0, Time elapsed: 0.512 s -- in net.ladenthin.llama.ffm.FfmNativeServerTest
[INFO] Tests run:  9, Failures: 0, Errors: 0, Skipped: 0, Time elapsed: 0.036 s -- in net.ladenthin.llama.ffm.JllamaNativeTest
[INFO] Tests run: 30, Failures: 0, Errors: 0, Skipped: 0
[INFO] BUILD SUCCESS
```

Der neue Test lief, aus dem surefire-XML belegt (eine Selbstabschaltung wäre hier der stille Fall,
den der CI-Job eigens prüft):

```
name="oaiCompletionStreamEndsWithUsage" classname="net.ladenthin.llama.ffm.FfmLlamaModelIntegrationTest" time="0.036"
```

Aufruf: `mvn -f llama-ffm/pom.xml verify
-Dnet.ladenthin.llama.ffm.model=models/AMD-Llama-135m-code.Q2_K.gguf
-Dnet.ladenthin.llama.lib.path=llama/src/main/natives/net/ladenthin/llama/Windows/x86_64/cpu`. Der
relative Modellpfad greift, weil `FfmTestSupport.modelPath()` auf `../models/…` zurückfällt.

**Folgerung:** Der FFM-Weg trägt mit dem echten Modell auf Windows, und F186 ist auch dort belegt:
`oaiCompletionStreamEndsWithUsage` verlangt `object: "text_completion"` auf jedem Chunk und ein
`usage.completion_tokens > 0` auf dem letzten. Für diesen Punkt musste ein JDK 22+ her — auf der
Maschine war keines, also wurde Temurin 25.0.4.1+1 (141 MB) installiert und ein
`~/.m2/toolchains.xml` mit 21 und 25 angelegt, den es vorher nicht gab.

---

## M10 Gesamtlauf

**Ziel.** Nach M1 … M9 einmal alles: `ctest` (erwartet 643/643) und
`mvn -B -f llama/pom.xml test -Dnet.ladenthin.llama.test.ngl=0` mit dem ganzen Modellsatz, dazu
`.github/verify-test-counts.sh` wie in CI. Zahlen (run / failures / errors / skipped) eintragen.

**Ergebnis:** Beide Hälften grün.

**`ctest`** gegen den statischen Build mit `-DBUILD_TESTING=ON` — die vom Auftrag erwartete Zahl:

```
100% tests passed, 0 tests failed out of 643
Total Test time (real) = 147.01 sec
```

**Die Java-Suite**, `mvn -f llama/pom.xml test -Dnet.ladenthin.llama.test.ngl=0` mit dem ganzen
Modellsatz:

```
[INFO] Tests run: 2151, Failures: 0, Errors: 0, Skipped: 18
[INFO] BUILD SUCCESS
[INFO] Total time:  08:48 min
```

**Die Zählprüfung wie in CI** (`.github/verify-test-counts.sh llama/target/surefire-reports
--min-executed 1800`):

```
test counts verified: 2151 test(s) run, 18 skipped, 2133 executed across 139 class(es), none empty (floors: total 0, executed 1800)
```

| | |
|---|---:|
| run | 2151 |
| failures | 0 |
| errors | 0 |
| skipped | 18 |
| executed (run − skipped) | 2133 |
| Klassen, keine leer | 139 |

**Eine Falle beim Zählen, die hier zuschlug und protokolliert gehört:** der erste Aufruf meldete
2158 über 146 Klassen. `mvn test` räumt `target/surefire-reports` nicht auf, also lagen dort noch die
XML der sieben Wegwerf-Proben aus M2 … M7 — genau 7 Klassen und 7 Tests mehr. Die Zahlen oben sind
nach dem Entfernen dieser Berichte, bestehen also nur aus dem Gesamtlauf. Wer die Zahl eines
Gesamtlaufs übernimmt, muss das Verzeichnis vorher leeren oder `clean` fahren.

**Die 18 Auslassungen, einzeln geprüft** — jede ist erklärt und für einen statischen Windows-Build
erwartet, keine ist eine stumme Klasse (`none empty` oben):

| Anzahl | Tests | Grund |
|---:|---|---|
| 8 | `BackendLoadTest` (alle) | `backend fixtures are committed for Linux only` |
| 2 | `RouterModeIntegrationTest#chatCompletion_isProxiedToWorker`, `#models_listContainsLoadedModel` | `Router worker-relaunch test runs on Linux only` |
| 2 | `ProcessRunnerTest#returnsStandardOutput`, `#commandThatDoesNotEndInTimeIsKilledAndReported` | `uname` bzw. `sleep` sind POSIX-Kommandos (`assumeFalse(WINDOWS, …)`) |
| 2 | `CpuModuleIsolationTest#everyCpuModuleLoadsOnItsOwn`, `LlamaLoaderTest#extractionDirectoriesOfThisClasspathNameTheLibraryWithItsModules` | brauchen die **modulare** Auslage aus den Jars; dieser Build ist der statische, den der Loader direkt aus `src/main/natives` lädt |
| 2 | `SystemOneIntegrationTest#aDecisionModelAnswersEveryQuestionType`, `#aDecisionModelRejectsAnUnknownQuestionType` | das Entscheidungsmodell ist nicht im CI-Satz (CLAUDE.md) |
| 1 | `AudioInputIntegrationTest#audioInputProducesNonEmptyReply` | Audio-Modell + mmproj sind nicht im CI-Satz (CLAUDE.md) |
| 1 | `LlamaModelTest#testLogStdout` | `@Disabled` |

**Folgerung:** Der Gesamtlauf bestätigt, dass die Messungen M1 … M9 den Baum so hinterlassen haben,
wie sie ihn gefunden haben: 643/643 in C++ und 2151/0/0/18 in Java, 2133 ausgeführt gegen eine
Untergrenze von 1800, keine Klasse mit null Einträgen.

### Nachlauf nach den Fixes

Die Zahlen oben sind der Gesamtlauf **vor** den Behebungen von B1/B2/B4 (B3 war zu diesem Zeitpunkt
schon drin). Danach wurde alles noch einmal gefahren, jedes Modul gegen den neu installierten Kern:

| Lauf | Ergebnis |
|---|---|
| `ctest` (C++, unverändert — die Fixes sind reines Java) | 643/643 |
| `mvn -f llama/pom.xml clean test` (ganzer Modellsatz) | **2154** / 0 / 0 / 18, 8:43 min |
| `verify-test-counts.sh … --min-executed 1800` | 2154 run, 18 skipped, **2136 executed**, 139 Klassen, keine leer |
| `spotbugs:check` wie im `code-style`-Job | `BugInstance size is 0`, `Error size is 0` |
| `spotless:check` (Kern und Agentenmodul) | grün |
| `llama-atmosphere-agent` (`verify` mit `lib.path`) | 281 / 0 / 0 / 60 |
| `llama-ffm` (`verify` mit echtem Modell) | 30 / 0 / 0 / 0 |
| `llama-langchain4j` (`verify`) | 55 / 0 / 0 / 7 |
| `llama-kotlin` (`verify`) | 6 / 0 / 0 / 0 |

2154 statt 2151 sind die drei neuen Tests in `ModelParametersTest` (Akkumulation, CSV-Quoting,
Doppelpunkt-Ablehnung); `ScriptedBackendTest` steckt in den 281 des Agentenmoduls. Die acht
Wegwerf-Proben der Messung und die zwei des Fix-Nachweises sind entfernt; was im Arbeitsbaum bleibt,
ist diese Datei plus die Fixes zu B1, B2, B3 und B4 mit ihren Wächtern.
