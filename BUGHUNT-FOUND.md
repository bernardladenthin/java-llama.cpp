<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT
-->


# Bug-Hunt: gefundene Fehler — java-llama.cpp

Fortlaufendes Fundbuch der Fehlersuche auf dem Arbeitsbranch `ccr-90fc5f6a-vs4s0f`. Jeder Fund hat eine
globale ID (`Fnn`, über alle drei Repos eindeutig), Datei:Zeile, Schwere, Beschreibung, Reproduktion (Testname),
Status und Commit. Neue Funde werden **unten angehängt**; die nächste freie ID ist die höchste ID in allen drei
Repos plus eins. Geprüfte Klassen ohne Fund stehen in [`BUGHUNT-CHECKED.md`](BUGHUNT-CHECKED.md).

## Übersicht

- [F01](#f01) — niedrig — *gefixt*
- [F05](#f05) — niedrig — *gefixt*
- [F06](#f06) — mittel — *gefixt*
- [F07](#f07) — niedrig — *gefixt*
- [F09](#f09) — hoch — *gefixt*
- [F12](#f12) — mittel — *gefixt*
- [F13](#f13) — mittel — *gefixt*
- [F16](#f16) — mittel — *gefixt*
- [F17](#f17) — mittel — *gefixt*
- [F18](#f18) — mittel — *gefixt*
- [F24](#f24) — mittel — *gefixt*
- [F25](#f25) — mittel — *gefixt*
- [F26](#f26) — mittel — *gefixt*
- [F28](#f28) — mittel — *gefixt*
- [F29](#f29) — mittel — *gefixt*
- [F30](#f30) — mittel — *gefixt*

## Funde

<a id="f01"></a>
### F01 – java-llama.cpp: `TrainingParameters` validiert die dokumentierten Grenzen nicht
- **Zeit:** 2026-10-16 22:50 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/parameters/TrainingParameters.java:46` (`epochs`, „at least 1“), `:19` (Klassen-Javadoc „exactly one of trainingText / trainingFile“), `valSplit`
- **Schwere:** niedrig
- **Beschreibung:** Javadoc (und `LlamaTrainer.finetune(Path,String,Path,int,float)`) sagt `epochs` „at least 1“ und „genau eins von trainingText/trainingFile“. Nichts davon wird geprüft: `epochs(0)` oder negativ läuft nativ 0 Epochen und `train_engine.cpp` schreibt danach das **unveränderte** Modell nach `outputPath` – ein stiller Erfolg ohne Training. Sind beide Korpus-Felder gesetzt, gewinnt nativ stillschweigend `trainingFile`. `valSplit` außerhalb `[0,1)` ergibt in `train_engine.cpp:75` einen negativen bzw. 0-großen `idata_split` (Verhalten in ggml nicht geprüft, möglicherweise ein `GGML_ASSERT`, der die JVM beendet). Fehlende `modelPath`/`outputPath` geben eine nackte NPE aus `toJson()`.
- **Reproduktion:** `TrainingParametersTest#lessThanOneEpochIsRejected`, `#validationSplitOutsideZeroToOneIsRejected`, `#bothOrNoCorpusIsRejected`, `#missingPathsNameTheField`
- **Fix:** `toJson()` ruft `validate()`: `modelPath`/`outputPath` per `Objects.requireNonNull` mit Feldnamen, genau eins von `trainingText`/`trainingFile`, `epochs >= 1`, `valSplit` in `[0, 1)` (NaN eingeschlossen) – sonst `IllegalArgumentException` vor dem JNI-Aufruf.
- **Rot-Grün-Nachweis:** ohne Fix rot, 4 von 7 („Expected IllegalArgumentException to be thrown, but nothing was thrown“ bzw. bare NPE ohne Feldnamen); mit Fix grün 7/7 (skipped 0), `WireNameRegistryTest` 8/8; Gegenprobe per `git stash` rot 4/7, nach `stash pop` grün 7/7. `WireNameRegistryTest#everyTrainingFieldIsWrittenByAFullyPopulatedConfiguration` setzte beide Korpus-Felder zugleich und nutzt jetzt zwei Konfigurationen.
- **Status:** gefixt
- **Commit:** java-llama.cpp `0aa7c94`

<a id="f05"></a>
### F05 – java-llama.cpp: `ChatRequest.buildToolsJson()` kürzt ein Tool-Schema still auf den ersten JSON-Wert
- **Zeit:** 2026-10-16 23:20 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/parameters/ChatRequest.java` (`buildToolsJson`, Zeile mit `SCHEMA_READER.readTree`)
- **Schwere:** niedrig
- **Beschreibung:** Das Schema jeder `ToolDefinition` wurde mit einfachem `readTree` gelesen; das liest den ersten JSON-Wert und ignoriert den Rest. `{"type":"object"} {"required":["x"]}` wurde so ohne Fehler als `{"type":"object"}` gesendet – ein anderes Schema als vom Aufrufer geschrieben. `JsonParameters` nutzt aus genau diesem Grund `FAIL_ON_TRAILING_TOKENS` (siehe CLAUDE.md), `ChatRequest` umging das. Jetzt gleicher strikter Reader → solche Strings nehmen den vorhandenen Fallback für ungültiges JSON.
- **Weitere Beobachtung (nicht geändert):** Ungültiges JSON wird absichtlich als **String** unter `parameters` weitergereicht, `""` und Leerzeichen werden zu `"parameters":null`. Ob eine `IllegalArgumentException` (wie in `JsonParameters`) besser wäre, ist eine API-Entscheidung.
- **Reproduktion:** `ChatRequestTest$…#buildToolsJsonDoesNotSilentlyTruncateASchemaWithTrailingContent` (vorher rot, danach grün), plus Positivtest `buildToolsJsonEmbedsAValidSchemaAsAnObject`; `ChatRequestTest` 24/24 grün. `spotbugs:check` nicht gelaufen.
- **Status:** gefixt
- **Commit:** java-llama.cpp `399ff0a`

<a id="f06"></a>
### F06 – java-llama.cpp: `RouterClient.awaitModelLoaded(id, Long.MAX_VALUE)` bricht sofort mit Timeout ab
- **Zeit:** 2026-10-16 23:35 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/RouterClient.java` (`awaitModelLoaded`, Deadline-Berechnung)
- **Schwere:** mittel
- **Beschreibung:** `System.currentTimeMillis() + timeoutMillis` läuft bei `Long.MAX_VALUE` („so lange wie nötig“, eine naheliegende Aufrufform) über und wird negativ. Die Schleife pollt nie und wirft „did not reach LOADED within 9223372036854775807 ms (last status: null)“, auch wenn das Modell längst geladen ist. Fix: Deadline sättigt bei `Long.MAX_VALUE`.
- **Reproduktion:** `RouterClientTest#awaitModelLoaded_longMaxValueTimeoutStillPolls` (vorher rot mit genau dieser Meldung, danach grün)
- **Status:** gefixt
- **Commit:** java-llama.cpp `ec12331`

<a id="f07"></a>
### F07 – java-llama.cpp: `RouterClient` mit IPv6-Literal (`"::1"`) kann nie verbinden
- **Zeit:** 2026-10-16 23:40 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/RouterClient.java` (`request`, URL-Aufbau)
- **Schwere:** niedrig
- **Beschreibung:** Die URL wurde als `"http://" + host + ":" + port` gebaut. Für `new RouterClient("::1", port)` (Router mit `--host ::1`; Javadoc: „host name or address“) entsteht `http://::1:port/models` → `MalformedURLException: Error at index 0 in ":1:8080"` vor jedem Verbindungsversuch. Workaround war `"[::1]"`. Fix: IPv6-Literal wird geklammert, bereits geklammerte Hosts bleiben unverändert. Nicht abgedeckt: Zonen-IDs (`fe80::1%eth0`) bräuchten zusätzlich `%25`-Kodierung. `toString()` zeigt den Host weiterhin ungeklammert.
- **Reproduktion:** `RouterClientTest#ipv6LiteralHostIsBracketedInTheRequestUrl` (vorher rot: `MalformedURLException`, danach andere `IOException`, weil auf Port 1 nichts lauscht). Der Sandbox fehlt IPv6, ein echter Server auf `::1` war hier nicht testbar.
- **Status:** gefixt
- **Commit:** java-llama.cpp `0f84984`

<a id="f09"></a>
### F09 – java-llama.cpp: Loader löscht fremde `llama*`/`ggml*`-Einträge im Temp-Verzeichnis, Verzeichnisse rekursiv
- **Zeit:** 2026-10-16 23:30 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/loader/LlamaLoader.java` (`isStaleExtraction` / `cleanup` / `cleanPath`)
- **Schwere:** hoch (Datenverlust außerhalb der eigenen Dateien)
- **Beschreibung:** `cleanup()` lief bei jedem Start, der die Natives-Jars erreicht, über `java.io.tmpdir` (bzw. `net.ladenthin.llama.tmpdir`) und löschte **sofort und rekursiv** jeden Eintrag, dessen Name mit `jllama`, `llama` oder `ggml` beginnt und kein `jllama-backend-*`-Verzeichnis ist. Gedacht war das für die einzelnen Bibliotheksdateien des flachen Layouts vor 6.0.0 (so steht es in der Javadoc). Getroffen wurde aber auch z. B. ein `/tmp/llama.cpp`-Checkout, ein `/tmp/ggml-models/`-Ordner oder ein heruntergeladenes `/tmp/llama-2-7b.Q4_K_M.gguf`. Fix: Aus dem alten Layout werden nur noch **reguläre Dateien mit Bibliotheksendung** gelöscht (`.so`, `.dll`, `.dylib`, `.jnilib`, `.metal`).
- **Reproduktion:** `LlamaLoaderTest#staleExtractionLeavesForeignEntriesWithAMatchingPrefixAlone` (vorher rot, danach grün); `LlamaLoaderTest` 63/63, inkl. der bestehenden Fälle `jllama-3f2a.so` / `ggml-metal.metal` (werden weiter gelöscht).
- **Bitte prüfen:** Die genauen Dateinamen des alten Layouts konnte ich nicht verifizieren, weil der Clone flach ist und keine Historie vor 6.0.0 enthält. Falls es damals auch Endungen wie `.tmp` oder `.lck` gab, bleiben die jetzt liegen. Das ist harmlos, nur Restmüll.
- **Status:** gefixt
- **Commit:** java-llama.cpp `3ae1b98`

<a id="f12"></a>
### F12 – java-llama.cpp: Ollama-`/api/chat` verwirft Bilder (`images`), obwohl `/api/show` „vision“ meldet
- **Zeit:** 2026-10-16 23:55 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OllamaApiSupport.java` (`toOpenAiMessage`)
- **Schwere:** mittel (Feature stillschweigend wirkungslos)
- **Beschreibung:** Ollama-Clients schicken Bilder als `"images": ["<base64>", …]` neben `"content"`. `toOpenAiMessage` kopierte die Nachricht unverändert; der OpenAI-Parser von llama.cpp liest nur `image_url`-Content-Parts, das Feld `images` wurde also ignoriert. Das Vision-Modell antwortete ohne Bild, kein Fehler. Gleichzeitig meldet `/api/show` die Fähigkeit `vision`, wenn `--mmproj` gesetzt ist. Fix: `images` wird entfernt und als `image_url`-Parts (Data-URL, MIME-Typ aus den Magic-Bytes: JPEG/GIF/WebP, sonst PNG – llama.cpp dekodiert nach Inhalt) hinter dem Text angehängt.
- **Reproduktion:** `OllamaApiSupportTest#toOpenAiChatRequestTurnsOllamaImagesIntoImageUrlParts` (vorher rot, danach grün); `OllamaApiSupportTest` 20/20, `OpenAiCompatServerHttpTest` 43/43. Ende-zu-Ende mit echtem Vision-Modell nicht geprüft (kein Modell im Sandbox).
- **Nebenbefund (nicht geändert):** Die gestreamte Abschlusszeile (`toOllamaDoneLine`) setzt `done_reason` immer auf `"stop"`, auch bei Abbruch durch `length`; die nicht-gestreamte Antwort übernimmt `finish_reason` korrekt.
- **Status:** gefixt
- **Commit:** java-llama.cpp `d31164b`

<a id="f13"></a>
### F13 – java-llama.cpp: `setRopeScaling(UNSPECIFIED | LONGROPE | MAX_VALUE)` macht das Modell unladbar
- **Zeit:** 2026-10-17 00:05 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/parameters/ModelParameters.java` (`setRopeScaling`), `args/RopeScalingType.java`
- **Schwere:** mittel
- **Beschreibung:** llama.cpp akzeptiert bei `--rope-scaling` nur `none`, `linear`, `yarn` (`common/arg.cpp` @ b11639, Zeile 2337ff.) und wirft sonst `invalid value` → `LlamaModel` lädt nicht („Failed to parse model parameters“). `RopeScalingType` bietet zusätzlich `UNSPECIFIED` (laut Javadoc „use the model default“), `LONGROPE` und `MAX_VALUE`; der Setter reichte sie ungeprüft als `unspecified`/`longrope`/`maxvalue` durch. Genau die Fehlerklasse aus CLAUDE.md („a Java test asserting the string mapping passes forever“): `testSetRopeScalingAllValues` prüfte nur die String-Abbildung aller Konstanten. Gegencheck aller anderen `args`-Enums gegen den Upstream-Parser (CacheType, DraftSampling, FlashAttn, GpuSplitMode, LazyMode, LoadMode, MiroStat, NumaStrategy, PoolingType, ReasoningFormat, Sampler): alle Werte akzeptiert.
- **Fix:** `UNSPECIFIED` entfernt die Option (wie `setPoolingType`), `LONGROPE`/`MAX_VALUE` → `IllegalArgumentException` beim Aufruf. Kein vorher funktionierender Pfad geht verloren (diese drei Werte scheiterten ohnehin beim Laden). Die Enum-Konstanten selbst bleiben (API-Kompatibilität) – ob sie gelöscht werden sollen, ist deine Entscheidung (CLAUDE.md: „A name with no counterpart is deleted, never deprecated“).
- **Reproduktion:** `ModelParametersExtendedTest#testSetRopeScalingUnspecifiedLeavesTheModelDefault`, `#testSetRopeScalingRejectsValuesTheParserDoesNotAccept` (beide vorher rot), `#testSetRopeScalingAllValuesTheParserAccepts` (ersetzt den alten Test). `WireNameRegistryTest`-Treiber nimmt bei Ablehnung die nächste Konstante. Grün: ModelParametersExtendedTest 148, ModelParametersTest 92, WireNameRegistryTest 8, RopeScalingTypeTest 24. Nicht gelaufen: C++-`test_model_flags` (prüft nur Optionsnamen, nicht Werte), `spotbugs:check`.
- **Status:** gefixt
- **Commit:** java-llama.cpp `8be77fe`

<a id="f16"></a>
### F16 – java-llama.cpp: `close()` parallel zu `synthesize()` gibt die TTS-Engine frei, während sie läuft (Use-after-free → JVM-Absturz)
- **Zeit:** 2026-10-17 00:50 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/TextToSpeech.java:95-113` (JNI) und `llama-ffm/src/main/java/net/ladenthin/llama/ffm/FfmTextToSpeech.java` (FFM); nativ `llama/src/main/cpp/tts_engine.cpp:201` (`engine_free` = `delete engine`), `jllama_api.cpp:673`
- **Schwere:** mittel (Fehlbedienung durch zwei Threads, aber Folge ist ein nativer Absturz statt einer Exception)
- **Beschreibung:** `synthesize()` prüft nur `handle == 0` bzw. `closed.get()` und ruft dann nativ auf, ohne Lock. `close()` (JNI: `synchronized`, FFM: CAS) ruft `jllama_tts_free` → `delete engine`. Schließt Thread B, während Thread A in `engine_synthesize` steckt, löscht B die Engine samt `synthesize_mutex`, `llama_context` und `mtmd`-Kontext unter A weg → Use-after-free im nativen Code. Bei `LlamaModel` gibt es dafür die Nutzer-Referenzzählung (`jllama_context_guard`, `free` wartet auf `users == 0`); die TTS-Engine hat nichts Vergleichbares. Mehrere gleichzeitige `synthesize()` sind dagegen sicher (nativ per `synthesize_mutex` serialisiert).
- **Vorschlag:** In beiden Klassen `synthesize` und `close` über denselben Monitor laufen lassen (z. B. `synchronized` auch auf `synthesize`; kostet nichts, weil nativ ohnehin serialisiert wird) – oder eine Referenzzählung wie bei `jllama_context` in `jllama_api.cpp`.
- **Reproduktion:** `TextToSpeechLifecycleTest#close_whileSynthesizeIsInsideTheEngine_freesItOnlyAfterSynthesizeReturned` (Fake über die neue paketinterne Naht `TextToSpeech.Backend` hält `synthesize` im „nativen“ Aufruf fest), dazu `#synthesize_afterClose_throwsIllegalState`, `#close_calledTwice_freesTheEngineOnce`, `#synthesize_whileOpen_returnsTheBackendsWav`
- **Fix:** `ReentrantReadWriteLock`: `synthesize` hält die Lesesperre über den ganzen nativen Aufruf, `close` die Schreibsperre. Gleichzeitige Synthesen bleiben möglich (die Engine ordnet sie selbst über `synthesize_mutex`), aber `close` gibt die Engine (samt diesem Mutex) erst frei, wenn keine mehr läuft. Die JNI-Aufrufe laufen über eine paketinterne Naht `Backend`; `LlamaLoader.initialize()` wanderte aus dem static-Block in deren `load`.
- **Rot-Grün-Nachweis:** Naht zuerst verhaltensneutral eingezogen; dagegen rot 1/4 („the engine was freed while synthesize() still ran in it – Expected: is <false> but: was <true>“). Mit Fix grün 4/4 (skipped 0); Gegenprobe mit dem Stand ohne Fix rot 1/4, wiederhergestellt grün 4/4. `LlamaArchitectureTest` 13/13, `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `128be3a`

<a id="f17"></a>
### F17 – java-llama.cpp: Responses-API (`POST /v1/responses`) verwirft `input_image`-Teile → Vision-Modell sieht kein Bild
- **Zeit:** 00:10 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/ResponsesApiSupport.java:138` (`appendInput`, Fall `message`)
- **Schwere:** mittel
- **Beschreibung:** Der Translator flachte jedes Content-Array einer `message` auf seinen Text ab (`inputContentText` liest nur `text`). Ein `{"type":"input_image","image_url":"data:…"}`-Teil kam nie im Chat-Kern an; ein Vision-Modell antwortete, als wäre kein Bild geschickt worden – still, ohne Fehler. Gleiche Fehlerklasse wie der Ollama-`images`-Fund oben.
- **Fix:** Enthält das Array `input_image`-Teile, wird ein OpenAI-Content-Part-Array gebaut (`text`-Teile + `image_url`-Teile; `image_url` als String oder `{url}`). Reine Text-Nachrichten bleiben unverändert ein String. `file_id`-Bilder (ohne URL) werden weiterhin ignoriert.
- **Reproduktion:** `ResponsesApiSupportTest#requestTurnsInputImagePartsIntoImageUrlParts` (vorher rot, nachher 10/10 grün)
- **Status:** gefixt
- **Commit:** java-llama.cpp `4c08a90`

<a id="f18"></a>
### F18 – java-llama.cpp: `NativeServer.close()` nicht atomar → doppeltes Freigeben des nativen Servers bei parallelen Aufrufen
- **Zeit:** 00:17 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/NativeServer.java:284` (`close`, ebenso `isRunning`, `start`)
- **Schwere:** mittel
- **Beschreibung:** `close()` las `handle`, prüfte auf 0 und setzte es danach auf 0 – ohne Synchronisation. Zwei gleichzeitige `close()` (Shutdown-Hook + try-with-resources, oder in `NativeServer.main` der Hook und der `finally`-Block, wenn der Worker genau beim Ctrl-C selbst endet) kamen beide durch und riefen `stopNativeServer(h)` doppelt: zweimal `join()` und zweimal `delete` desselben `native_server` → JVM-Absturz. Ebenso konnte `isRunning()` einen Handle abfragen, den ein paralleles `close()` gerade freigab (Use-after-free). Der Kommentar in `main` („the finally and the hook both firing is safe“) gilt nur nacheinander.
- **Fix:** privates `lifecycleLock` serialisiert `start`, `isRunning` und `close` (keine `synchronized`-Methoden, SpotBugs USO_UNSAFE_METHOD_SYNCHRONIZATION). `spotbugs:check` grün, `NativeServerSmokeTest` + `LlamaArchitectureTest` grün.
- **Reproduktion:** `NativeServerLifecycleTest` über die neue paketinterne Naht `NativeServer.Backend` (ein Fake statt der JNI-Aufrufe, damit kein erfundener Handle libjllama erreicht): `close_whileIsRunningIsInsideTheNativeCall_stopsOnlyAfterItReturned`, `close_whileStartIsStillStarting_stopsTheServerOnceItStarted`, `close_calledTwice_stopsTheHandleOnce`. Der Fake hält `isRunning` bzw. `start` im „nativen“ Aufruf fest, dadurch sind beide Race-Fenster deterministisch. (Die alte Probe `/home/user/BUGHUNT-2026-10-17-F18-probe.java`: vorher 4 / 20 000 doppelte Stopps, nachher 0 / 20 000.)
- **Rot-Grün-Nachweis:** Gegen den Lebenszyklus vor `c1e5dcd` (ohne `lifecycleLock`, Naht beibehalten) **rot**, 2 von 3: „the handle was stopped while isRunning() still used it – Expected: is <false> but: was <true>“ und „close() returned while the server was still starting, so it kept running – Expected: is <1> but: was <0>“. Mit Fix **grün** 3/3 (skipped 0), dreimal hintereinander, ebenso `NativeServerSmokeTest` 10/10, `NativeServerWorkerCommandTest` 5/5, `LlamaArchitectureTest` 13/13, `spotbugs:check` grün. Das doppelte `close()` aus zwei Threads selbst ist mit der Naht nicht steuerbar (das Fenster liegt zwischen Lesen und Nullsetzen des Feldes); `close_calledTwice_stopsTheHandleOnce` deckt nur den sequentiellen Fall ab.
- **Status:** gefixt (Fix `c1e5dcd`; roter Regressionstest nachgereicht, bis dahin „nur beschrieben“, siehe `209ee6c`)
- **Commit:** java-llama.cpp `c1e5dcd` (Fix), Test und Naht `669a472`

<a id="f24"></a>
### F24 – java-llama.cpp: `OpenAiServerCli` bewirbt bei kleinem `-c` fast kein Eingabebudget (`-c 2048` → 1 Token)
- **Zeit:** 05:55 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiServerCli.java:721` (`Options.toServerConfig`)
- **Schwere:** mittel
- **Beschreibung:** Das Ausgabebudget ist `min(2048, ctx/2)`, das Eingabebudget wurde aber als `ctx − 2048` (der Default, nicht das tatsächliche Ausgabebudget) berechnet. Für `ctx < 4096` passt das nicht zusammen: `-c 2048` → Eingabe **1** Token, Ausgabe 1024; `-c 1024` → beworbenes Fenster 513 statt 1024; `-c 3000` → 2452. Das Eingabebudget landet in `maxInputTokens` (Copilot-Konfigurationsausgabe) und in der beworbenen Kontextlänge (`/api/show`, `/props`: `maxInputTokens + maxOutputTokens`); Clients, die ihren Prompt danach zuschneiden, schicken praktisch nichts. Für `-c ≥ 4096` (README-Beispiel 8192 → 6144) war das Ergebnis zufällig richtig.
- **Fix:** `maxInputTokens = max(1, ctx − maxOutputTokens)` mit dem tatsächlich gesetzten Ausgabebudget; 8192 → 2048/6144 unverändert.
- **Reproduktion:** `OpenAiServerCliTest#smallContextSizeSplitsTheWholeWindowBetweenInputAndOutput` (ctx 1024/2048/3000; vorher `Expected 1024 but was 513`, nachher 38/38 grün).
- **Status:** gefixt
- **Commit:** java-llama.cpp `b210ca4`

<a id="f25"></a>
### F25 – java-llama.cpp: `OpenAiCompatServer` mit `--api-key` lehnt jeden Anthropic-Client ab (`x-api-key`) und `bearer` in Kleinschreibung
- **Zeit:** 06:03 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiCompatServer.java:893` (`authorized`; gefunden beim Review von `OpenAiServerConfig`)
- **Schwere:** mittel
- **Beschreibung:** Akzeptiert wurde nur `Authorization: Bearer <key>` mit exakt großgeschriebenem `Bearer`. Die Anthropic-Messages-Oberfläche (`POST /v1/messages`) wird von Anthropic-SDKs aber per `x-api-key`-Header authentifiziert – mit gesetztem `--api-key` bekam jeder solche Client 401, die Anthropic-Oberfläche war damit nur ohne Schlüssel nutzbar. Zudem ist der Schema-Name laut RFC 7235 case-insensitiv; `bearer <key>` wurde abgelehnt.
- **Fix:** `x-api-key` wird zusätzlich akzeptiert, das Schema case-insensitiv geprüft; beide Vergleiche weiterhin zeitkonstant (`MessageDigest.isEqual`). `spotbugs:check` grün.
- **Reproduktion:** `OpenAiCompatServerHttpTest#anthropicApiKeyHeaderAndLowercaseBearerSchemeAreAccepted` (vorher 401 statt 200, nachher Klasse 44/44 grün).
- **Status:** gefixt
- **Commit:** java-llama.cpp `e98e290`

<a id="f26"></a>
### F26 – java-llama.cpp: `withTokenIdBias`/`withTokenBias` mit `Float.NEGATIVE_INFINITY` sperrt das Token nicht – wird still ignoriert
- **Zeit:** 06:09 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/parameters/ParameterJsonSerializer.java:236` (`buildTokenIdBiasArray`/`buildTokenStringBiasArray`; gefunden beim Review von `InferenceParameters`)
- **Schwere:** mittel
- **Beschreibung:** JSON kennt kein Unendlich; Jackson schreibt einen nicht-endlichen `float` als String: `{"logit_bias":[[5,"-Infinity"]]}` (gemessen). llama.cpps Parser (`server-schema.cpp` b11639, `parse_bias`) akzeptiert nur eine Zahl oder `false` und überspringt alles andere per `continue`. Der naheliegende Weg, ein Token zu verbieten (`withTokenIdBias(Map.of(id, Float.NEGATIVE_INFINITY))`), bewirkte damit gar nichts – ohne Fehler. `NaN` und `+Infinity` ebenso.
- **Fix:** `-Infinity` wird als `false` (llama.cpps Schreibweise für ein Verbot, wie schon bei `withDisabledTokenIds`) gesendet, `NaN`/`+Infinity` werfen `IllegalArgumentException`.
- **Reproduktion:** `ParameterJsonSerializerTest#testBuildTokenIdBiasArray_negativeInfinityBansTheToken`, `#testBuildTokenIdBiasArray_nanOrPositiveInfinityIsRejected` (vorher rot, nachher 34/34); `InferenceParametersTest` 105/105, `WireNameRegistryTest` 8/8 grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `ea721a2`

<a id="f28"></a>
### F28 – java-llama.cpp: `InferenceParameters.withMessages(List<ChatMessage>)` verwirft `tool_calls` und `tool_call_id`
- **Zeit:** 06:22 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/parameters/ParameterJsonSerializer.java:152` (`buildMessages(List<ChatMessage>)`)
- **Schwere:** mittel
- **Beschreibung:** Der gemeinsame Serializer schrieb nur `role` und `content` (bzw. Content-Parts). `ChatRequest.buildMessagesJson` ergänzte `tool_calls`/`tool_call_id` nachträglich selbst – `InferenceParameters.withMessages(List<ChatMessage>)` nutzt den Serializer aber direkt. Eine Tool-Calling-Runde über diesen Weg (`ChatMessage.assistantToolCalls(...)`, `ChatMessage.toolResult(...)`) schickte einen Assistant-Turn ohne seine Aufrufe und ein Tool-Ergebnis ohne Zuordnung: `[{"role":"assistant","content":""},{"role":"tool","content":"42"}]` (gemessen).
- **Fix:** die beiden Felder schreibt jetzt der Serializer; die Duplikat-Schleife in `ChatRequest` entfällt (Ausgabe dort identisch). `spotbugs:check` grün.
- **Reproduktion:** `InferenceParametersTest#withChatMessagesKeepsToolCallsAndToolCallIds` (vorher rot, nachher 106/106); `ChatRequestTest` 24/24, `ParameterJsonSerializerTest` 34/34, `OpenAiCompatServerHttpTest` 44/44 grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `855d22e`

<a id="f29"></a>
### F29 – java-llama.cpp: Anthropic-Messages-API (`POST /v1/messages`) verwirft `image`-Blöcke → Vision-Modell sieht kein Bild
- **Zeit:** 06:29 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/AnthropicApiSupport.java:151` (`appendOpenAiMessages`)
- **Schwere:** mittel
- **Beschreibung:** Gleiche Fehlerklasse wie F17 (Responses) und der Ollama-`images`-Fund: der Translator kannte nur `text`, `tool_use` und `tool_result`; ein `{"type":"image","source":{"type":"base64","media_type":…,"data":…}}`-Block (so senden Anthropic-Clients Bilder) fiel im `default`-Zweig weg. Ein Vision-Modell hinter `/v1/messages` antwortete, als wäre kein Bild geschickt worden – still.
- **Fix:** Bild-Blöcke werden zu OpenAI-`image_url`-Teilen (Data-URL für `base64`, die URL selbst für `url`-Quellen); eine Nachricht mit Bild bekommt Content als Teile-Array in Blockreihenfolge, Tool-Calls bleiben erhalten. Nachrichten ohne Bild unverändert. `spotbugs:check` grün.
- **Reproduktion:** `AnthropicApiSupportTest#requestTurnsImageBlocksIntoImageUrlParts` (vorher rot, nachher 12/12); `OpenAiCompatServerHttpTest` 44/44 grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `d8c653c`

<a id="f30"></a>
### F30 – java-llama.cpp: Fehlgeschlagener `RpcServer.start` lässt den Server weiterlaufen, wenn der Stopp vor dem Bind kommt
- **Zeit:** 06:34 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/RpcServer.java:526` (vorher: einmaliges `backend.stop()` in `awaitListening`)
- **Schwere:** mittel
- **Beschreibung:** Läuft der Start in den Timeout (z. B. langsame Geräte-Initialisierung) oder ist der aufrufende Thread unterbrochen, schickt `start` genau **einen** `stop()` und gibt den Single-Instance-Slot frei. Der native Server (patches/0015) setzt aber beim Binden des Sockets `g_rpc_server_stop = false` – ein Stopp vor dem Bind geht verloren. Der Server lauscht danach als verwaister Daemon-Thread weiter, obwohl `start` eine `LlamaException` geworfen hat; ein zweiter `start` belegt dann denselben globalen Server-Zustand. Beim unterbrochenen Aufrufer kehrte `join` zudem sofort zurück, statt zu warten. Fix: `stopAndJoin` wiederholt den Stopp alle 100 ms, solange der Thread lebt (bis `STOP_TIMEOUT_MILLIS`), und stellt das Interrupt-Flag danach wieder her.
- **Reproduktion:** `RpcServerLifecycleTest#aStopThatArrivesBeforeTheBindDoesNotLeaveTheServerRunning` (Fake-Backend, das wie der native Server das Stopp-Flag beim Binden zurücksetzt; ohne Fix rot nach 31 s, mit Fix 15/15 grün)
- **Status:** gefixt
- **Commit:** 32d465d

## Offene Verdachte ohne Fund-ID

- *(unsicher/niedrig, nur beschrieben, 06:38 UTC)* java-llama.cpp `llama-ffm/.../ffm/JllamaNative.java` `takeStrings`: Bei `count == 0` kehrt die Methode vor dem `try` zurück und gibt das Array nicht frei, das `jllama_rpc_server_devices` auch für 0 Einträge per `calloc(max(n,1))` anlegt (8 Byte Leck je Aufruf). `rpc::server_devices()` fällt aber immer auf das CPU-Gerät zurück, leer wird die Liste nur ohne registriertes CPU-Backend (kaputte Installation) – praktisch unerreichbar. Fix wäre: bei Nicht-NULL-Zeiger immer `STRINGS_FREE` aufrufen. Nicht ohne native Bibliothek + JDK-22-Toolchain testbar, daher kein Commit. Alle übrigen Freigabe-Funktionen (`takeBytesAsString` → `jllama_string_free` laut Header) stimmen mit `jllama_api.h` überein. **Blocker (Stand 2026-10-17):** Der Rot-Grün-Zyklus braucht ein JDK 22+ (Toolchain des Moduls) und ist in dieser Sandbox nicht herstellbar: GitHub-Releases (Adoptium) und api.adoptium.net sind über den Proxy gesperrt. Vorschlag für den Test: `takeStrings` mit der Freigabefunktion als Parameter (paketintern), Aufruf mit Nicht-NULL-Zeiger und Anzahl 0, Erwartung genau ein Freigabeaufruf; läuft dann im CI-Job `test-java-llama-ffm`.
