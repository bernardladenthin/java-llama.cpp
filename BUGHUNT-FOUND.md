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
- [F36](#f36) — mittel — *gefixt*
- [F37](#f37) — niedrig — *gefixt*
- [F38](#f38) — mittel — *gefixt*
- [F39](#f39) — niedrig — *gefixt*
- [F53](#f53) — niedrig — *gefixt*
- [F54](#f54) — niedrig — *gefixt*
- [F55](#f55) — niedrig — *gefixt*
- [F56](#f56) — niedrig — *gefixt*
- [F57](#f57) — niedrig — *nur beschrieben* (Entscheidung des Nutzers)
- [F59](#f59) — niedrig — *nur beschrieben* (CI-Bedarf: JDK 22+)
- [F60](#f60) — niedrig — *gefixt*
- [F61](#f61) — niedrig — *gefixt*
- [F62](#f62) — niedrig — *nur beschrieben* (Entscheidung des Nutzers)
- [F63](#f63) — niedrig — *gefixt*
- [F64](#f64) — mittel — *gefixt*
- [F65](#f65) — niedrig — *gefixt*
- [F67](#f67) — niedrig — *nur beschrieben* (CI-Bedarf: Modell + LoRA-Adapter)
- [F68](#f68) — niedrig — *gefixt*
- [F69](#f69) — niedrig — *nur beschrieben* (CI-Bedarf: modellgestützter Test)
- [F70](#f70) — mittel — *gefixt*
- [F71](#f71) — niedrig — *gefixt*
- [F72](#f72) — niedrig — *gefixt*
- [F74](#f74) — mittel — *gefixt*
- [F75](#f75) — mittel — *gefixt*
- [F76](#f76) — mittel — *gefixt*
- [F77](#f77) — mittel — *gefixt*
- [F78](#f78) — mittel — *gefixt*
- [F79](#f79) — niedrig — *gefixt*
- [F80](#f80) — niedrig — *gefixt*
- [F81](#f81) — niedrig — *gefixt*
- [F82](#f82) — niedrig — *gefixt*
- [F83](#f83) — niedrig — *gefixt*
- [F84](#f84) — niedrig — *gefixt*
- [F85](#f85) — mittel — *gefixt*
- [F86](#f86) — mittel — *gefixt*
- [F87](#f87) — niedrig — *gefixt*
- [F88](#f88) — mittel — *gefixt*
- [F89](#f89) — niedrig — *gefixt*
- [F90](#f90) — mittel — *gefixt*
- [F91](#f91) — niedrig — *gefixt*
- [F92](#f92) — niedrig — *gefixt*
- [F93](#f93) — niedrig — *gefixt*
- [F94](#f94) — niedrig — *gefixt*
- [F95](#f95) — mittel — *gefixt*
- [F96](#f96) — niedrig — *gefixt*
- [F97](#f97) — niedrig — *gefixt*
- [F98](#f98) — niedrig — *gefixt*
- [F99](#f99) — niedrig — *gefixt*
- [F102](#f102) — niedrig — *gefixt*
- [F103](#f103) — niedrig — *gefixt*
- [F104](#f104) — niedrig — *gefixt*
- [F105](#f105) — mittel — *gefixt*
- [F106](#f106) — niedrig — *gefixt*
- [F107](#f107) — niedrig — *gefixt*
- [F108](#f108) — niedrig — *gefixt*
- [F109](#f109) — niedrig — *gefixt*
- [F111](#f111) — niedrig — *gefixt*
- [F112](#f112) — mittel — *gefixt*
- [F113](#f113) — niedrig — *gefixt*
- [F114](#f114) — mittel — *gefixt*
- [F115](#f115) — niedrig — *gefixt*

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
- **Doku-Lücke aus diesem Fix (2026-10-17, nur beschrieben):** Seit dem Fix wirft `TrainingParameters.toJson()` – und damit beide `LlamaTrainer.finetune`-Überladungen – bei ungültiger Konfiguration `IllegalArgumentException`; die Javadoc beider `finetune` nennt nur `LlamaException`. Reine Doku, kein Test kann sie rot machen, daher nach der Nutzerregel unverändert.

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

<a id="f36"></a>
### F36 – java-llama.cpp: `LlamaIterator` wird nach `cancel()` aus einem zweiten Thread durch ein spät eintreffendes Ergebnis wiederbelebt
- **Zeit:** 2026-10-17 16:40 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/LlamaIterator.java:85` (`next`)
- **Schwere:** mittel
- **Beschreibung:** `next()` setzte nach jedem Ergebnis `hasNext = !output.stop`. Ruft ein anderer Thread `cancel()` auf, während `next()` in `receiveCompletionJson` auf das nächste Ergebnis wartet (das Feld ist genau dafür `volatile`, und die Slot-Arbeit läuft nach `cancel()` weiter), dann überschreibt das danach eintreffende Nicht-Stopp-Ergebnis das `false` von `cancel()` wieder mit `true`. Das Iterator „lebt“ wieder, `hasNext()` meldet weitere Tokens, und der nächste `next()` scheitert mit „Task not found“, weil `cancel()` den Reader schon freigegeben hat – entgegen dem Javadoc („the iterator stops yielding tokens immediately“). Betrifft `generate`/`generateChat` und jede `LlamaIterable`, die aus einem zweiten Thread abgebrochen wird.
- **Fix:** `hasNext` geht nur noch von `true` nach `false`; `next()` schreibt `false` nur beim Stopp-Ergebnis und nie `true`. Für den Test bekommt der Iterator eine paketinterne Naht `LlamaIterator.TaskSource` (die fünf Task-Aufrufe, Adapter `TaskSource.of(LlamaModel)`); die JNI-Methoden bleiben in `LlamaModel`.
- **Reproduktion:** `LlamaIteratorCancelTest#cancel_whileNextWaitsForAResultOnAnotherThread_iteratorStaysFinished` (der Fake hält `receiveCompletionJson` fest, bis `cancel()` gelaufen ist), dazu `#next_untilTheStopResult_releasesTheTaskOnceAndEnds`, `#close_beforeTheStopResult_cancelsTheTaskOnce`
- **Rot-Grün-Nachweis:** Naht zuerst verhaltensneutral; dagegen rot 1/3 („a result that arrived after cancel() revived the iterator – Expected: is <false> but: was <true>“). Mit Fix grün 3/3 (skipped 0); Gegenprobe mit dem Stand ohne Fix rot, wiederhergestellt grün. `LlamaArchitectureTest` 13/13, `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `f4407bb`

<a id="f37"></a>
### F37 – java-llama.cpp: Ollama-Tool-Call mit leeren Argumenten kommt als `"arguments": null` statt `{}` an
- **Zeit:** 2026-10-17 17:00 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OllamaApiSupport.java:345` (`parseArgumentsToObject`, genutzt von `toOllamaToolCalls` für `/api/chat` gestreamt und blockierend)
- **Schwere:** niedrig
- **Beschreibung:** Ollama erwartet `tool_calls[].function.arguments` als Objekt. Die Umwandlung aus dem OpenAI-String parste mit `readTree` und fiel nur bei einer `IOException` auf `{}` zurück (so der Kommentar). `readTree("")` wirft aber nicht, sondern liefert einen `MissingNode`, der als `null` serialisiert wird; ebenso ergab ein skalarer Text (`"5"`) einen Skalar. Ein parameterloser Aufruf, für den llama.cpp `"arguments": ""` liefert (oder dessen gestreamte Deltas keine Argumente enthalten, `ToolCallDeltaAccumulator` setzt dann `""`), kam bei Ollama-Clients als `"arguments": null` an. Fix: alles, was nicht zu einem Objekt oder Array parst, wird `{}`.
- **Reproduktion:** `OllamaApiSupportTest#toOllamaChatResponseTurnsEmptyToolCallArgumentsIntoAnEmptyObject` (Jackson-Verhalten vorab per Probe bestätigt: `readTree("")` → `MissingNode` → `"arguments":null`)
- **Rot-Grün-Nachweis:** ohne Fix rot 1/21 („Expected: is <true> but: was <false>“: kein Objekt), mit Fix grün 21/21 (skipped 0); Gegenprobe per `git stash` rot, nach `stash pop` grün. `OpenAiCompatServerHttpTest` 44/44, `LlamaArchitectureTest` 13/13, `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `6590af2`

<a id="f38"></a>
### F38 – java-llama.cpp: `JsonParameters` nimmt einen leeren oder nur aus Leerzeichen bestehenden Wert still an und sendet `"key":null`
- **Zeit:** 2026-10-17 17:10 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/parameters/JsonParameters.java:181` (`parseOrThrow`)
- **Schwere:** mittel
- **Beschreibung:** `JsonParameters` garantiert laut CLAUDE.md, dass jeder gespeicherte Wert genau ein wohlgeformter JSON-Wert ist, geprüft beim Schreiben. Die Prüfung verließ sich darauf, dass `readTree` bei ungültiger Eingabe wirft. Für eine leere oder nur aus Leerzeichen bestehende Eingabe wirft auch der strikte Reader (`FAIL_ON_TRAILING_TOKENS`) nicht, sondern liefert einen `MissingNode` (per Probe bestätigt), der im Request als `"key":null` landete. `withJsonSchema("")`, `withResponseFormat("")`, `withToolsJson("")` usw. wurden so still angenommen, und die Generierung lief zum Beispiel ohne das Schema, das der Aufrufer zu setzen glaubte. Gefunden über die Erkenntnis aus F37 (`readTree("")` → `MissingNode`).
- **Fix:** `parseOrThrow` weist einen `MissingNode` mit derselben `IllegalArgumentException` ab wie jeden anderen ungültigen Wert; die Meldung baut ein gemeinsamer Helfer `rejected(...)`.
- **Reproduktion:** `JsonParametersTest#withRaw_rejectsEmptyAndBlankValues`
- **Rot-Grün-Nachweis:** ohne Fix rot 1/25 („Expected java.lang.IllegalArgumentException to be thrown, but nothing was thrown.“), mit Fix grün 25/25 (skipped 0); Gegenprobe per `git stash` rot, nach `stash pop` grün. Paket `parameters` plus Server-Mapping- und HTTP-Tests: 775 Tests, 0 Fehler. PIT auf `JsonParameters` (Schwelle 100): alle Mutanten getötet. `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `41b5ad3`

<a id="f39"></a>
### F39 – java-llama.cpp: Anthropic-`tool_use` mit leeren Argumenten bekommt `"input": null` statt `{}`
- **Zeit:** 2026-10-17 17:15 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/AnthropicApiSupport.java:331` (`parseToObject`, genutzt für `tool_use.input` in `toAnthropicResponse`)
- **Schwere:** niedrig
- **Beschreibung:** Dasselbe Muster wie F37, gefunden in der `readTree`-Durchsicht nach F37: Die blockierende Antwort von `POST /v1/messages` wandelte die OpenAI-Argumente per `readTree` in `tool_use.input` und fiel nur bei einer `IOException` auf `{}` zurück. Für `"arguments": ""` (parameterloser Aufruf) entstand ein `MissingNode`, also `"input": null`; Anthropic verlangt dort ein Objekt, Clients wie das Anthropic-SDK lesen `input` als Dict. Fix: nur Objekt/Array durchreichen, sonst `{}`.
- **Reproduktion:** `AnthropicApiSupportTest#toAnthropicResponseTurnsEmptyToolCallArgumentsIntoAnEmptyInputObject`
- **Rot-Grün-Nachweis:** ohne Fix rot 1/13 („Expected: is <true> but: was <false>“), mit Fix grün 13/13 (skipped 0); Gegenprobe per `git stash` rot, danach grün. `AnthropicStreamTranslatorTest` 4/4, `OpenAiCompatServerHttpTest` 44/44, `LlamaArchitectureTest` 13/13, `spotbugs:check` grün.
- **Durchsicht der übrigen `readTree`-Aufrufer (2026-10-17):** 35 Aufrufe in 24 Dateien. Betroffen waren nur F37 (Ollama), F38 (`JsonParameters`) und F39. Die übrigen lesen Antworten des eigenen Servers (Completion-/Chat-/Stream-Parser, Rerank, LoRA, Router) und greifen danach über `path(...)` zu, was auf einem `MissingNode` sicher leere/Default-Werte liefert; `ChatRequest.buildToolsJson` liest ein Tool-Schema, das ein leerer String zu `null` macht (bereits als F05-Umfeld bekannt), `StreamingChunkAssembler` wirft bei unparsebarem Chunk bewusst.
- **Status:** gefixt
- **Commit:** java-llama.cpp `6effd44`

<a id="f53"></a>
### F53 – java-llama.cpp: leerer Streaming-Umschlag beendet den Chat-Stream nicht
- **Zeit:** 2026-10-17 21:00 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/json/ChatStreamChunkParser.java:58` (`feed`)
- **Schwere:** niedrig
- **Beschreibung:** Die Javadoc verspricht: „An unparseable envelope is treated as end-of-stream (returns true) so a polling loop cannot spin forever on malformed input.“ Für Nicht-JSON stimmt das (`IOException` → `true`), für einen leeren oder nur aus Leerraum bestehenden Umschlag nicht: `readTree` liefert dann `MissingNode` statt zu werfen (dieselbe Falle wie F37–F39), `path("stop")` ergibt `false`, und `LlamaModel`s Pollschleife (`chatStreamParser.feed` bei Zeile 823) läse weiter.
- **Fix:** Nach dem Parsen gilt auch `root.isMissingNode()` als Streamende.
- **Reproduktion:** `ChatStreamChunkParserTest#feed_emptyOrBlankEnvelope_reportsStop_andEmitsNothing` (`""`, `"   "`, `"\n"`)
- **Rot-Grün-Nachweis:** ohne Fix rot („envelope '' Expected: is <true> but: was <false>“), mit Fix grün; Gegenprobe per `git stash` rot, nach `stash pop` grün. `ChatStreamChunkParserTest` 7/7, `LlamaArchitectureTest` 13/13, `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `27d6477`

<a id="f54"></a>
### F54 – java-llama.cpp: leerer Wert in `chat_template_kwargs` wird zu `null` statt zum Text
- **Zeit:** 2026-10-17 21:15 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/parameters/ParameterJsonSerializer.java:316` (`buildRawValueObject`)
- **Schwere:** niedrig
- **Beschreibung:** Erneute `readTree`-Durchsicht nach F53, jede Fundstelle mit `""` durchgespielt. Die Javadoc verspricht: „If a value cannot be parsed as JSON, it is stored as a JSON string literal.“ Für `"not json"` stimmt das, für einen leeren oder nur aus Leerraum bestehenden Wert nicht: `readTree` liefert `MissingNode`, und ein `MissingNode` als Kind eines `ObjectNode` wird als `null` geschrieben (in der Sandbox geprüft). `ModelParameters.setChatTemplateKwargs` und `InferenceParameters.withChatTemplateKwargs` gaben so `{"x":null}` an die Jinja-Vorlage weiter statt `{"x":""}`.
- **Fix:** `MissingNode` wird wie ein Parsefehler behandelt und der Text unverändert als String eingetragen.
- **Reproduktion:** `ParameterJsonSerializerTest#testBuildRawValueObject_emptyOrBlankValueFallsBackToString`
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: is "{"empty":"","blank":"   "}" but: was "{"empty":null,"blank":null}"“), mit Fix grün; Gegenprobe per `git stash` rot, nach `stash pop` grün. `ParameterJsonSerializerTest` 35/35, `ModelParametersTest` 92/92, `InferenceParametersTest` 106/106, `WireNameRegistryTest` 8/8, `LlamaArchitectureTest` 13/13, `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `37303d0`

<a id="f55"></a>
### F55 – java-llama.cpp: leeres Tool-Parameterschema wird zu `"parameters": null`
- **Zeit:** 2026-10-17 21:30 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/parameters/ChatRequest.java:329` (`buildToolsJson`)
- **Schwere:** niedrig
- **Beschreibung:** Aus der `readTree`-Durchsicht nach F53/F54. Ein nicht parsbares Parameterschema eines `ToolDefinition` wird bewusst als String weitergereicht (Test `buildToolsJsonDoesNotSilentlyTruncateASchemaWithTrailingContent`), damit es auffällt statt still verändert zu werden. Ein leeres oder nur aus Leerraum bestehendes Schema ergab dagegen `MissingNode` und damit `"parameters": null` – das Werkzeug ginge ohne Schema an die native Seite.
- **Fix:** `MissingNode` nimmt denselben String-Rückfall wie jedes andere nicht parsbare Schema.
- **Reproduktion:** `ChatRequestTest#buildToolsJsonKeepsAnEmptySchemaAsTheStringFallbackInsteadOfNull`
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: a string containing "\"parameters\":\"\"" but: was "[{"type":"function","function":{"name":"f","description":"d","parameters":null}}]"“), mit Fix grün; Gegenprobe per `git stash` rot, nach `stash pop` grün. `ChatRequestTest` 25/25, `ToolCallingAgentTest` 6/6, `LlamaArchitectureTest` 13/13, `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `ed6236d`

### F56 – java-llama.cpp: nicht darstellbarer Offline-Modellpfad ergibt `InvalidPathException` statt `ModelUnavailableException`
- **Zeit:** 2026-10-17 22:10 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/loader/OfflineModelGuard.java:62` (`check`)
- **Schwere:** niedrig
- **Beschreibung:** Die Javadoc verspricht eine `ModelUnavailableException`, wenn `--offline` gesetzt ist und die konfigurierte lokale Modelldatei fehlt. Ein Pfad, den das Dateisystem nicht darstellen kann (NUL-Zeichen; unter Windows auch `?`, `*`, `<` …), ließ `Paths.get` eine `InvalidPathException` werfen – eine untypisierte `IllegalArgumentException`, noch bevor der native Lader erreicht wird. Ein solcher Pfad kann keine Datei bezeichnen, ist also genau der Fall „Datei fehlt“.
- **Fix:** `localFileExists` behandelt eine `InvalidPathException` als „existiert nicht“.
- **Reproduktion:** `LlamaModelOfflineTest#check_offlineAndUnrepresentablePath_throwsModelUnavailable`
- **Rot-Grün-Nachweis:** ohne Fix rot („expected: <ModelUnavailableException> but was: <java.nio.file.InvalidPathException>“, Nul character not allowed), mit Fix grün; Gegenprobe per `git stash` rot, nach `stash pop` grün. `LlamaModelOfflineTest` 9/9, `LlamaArchitectureTest` 13/13, `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `4f610bd`

<a id="f57"></a>
### F57 – java-llama.cpp: `ServerMetrics.getCumulativeUsage()` liefert ein `Usage` gegen dessen eigenen Vertrag
- **Zeit:** 2026-10-17 21:05 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/value/ServerMetrics.java:203` (`getCumulativeUsage`)
- **Schwere:** niedrig
- **Beschreibung:** `Usage.getPromptTokens()` ist dokumentiert als „logical prompt token count, including cached tokens“, `getCachedTokens()` als dessen gecachte Teilmenge. `getCumulativeUsage()` füllt `promptTokens` aber mit `n_prompt_tokens_processed_total` (nur ausgewertete Tokens, ohne Cache-Treffer) und `cachedTokens` mit 0. Die Javadoc der Methode sagt das zwar, begründet es aber mit „upstream does not currently expose a cumulative logical or cached prompt total“ – das stimmt nicht mehr: `n_prompt_tokens_cached_total` kommt seit der `getMetrics()`-Zusammenführung mit und hat einen eigenen Getter (`getCumulativeCachedPromptTokens`). Ein Aufrufer, der `Usage` wie bei einer Chat-Antwort liest (`getPromptTokens()` gesamt, `getProcessedPromptTokens()` ausgewertet), bekommt hier zu kleine Gesamtzahlen.
- **Fix (Vorschlag, nicht umgesetzt):** `new Usage(processed + cached, predicted, cached)`; dann stimmen `getPromptTokens()`, `getCachedTokens()` und `getProcessedPromptTokens()` mit dem `Usage`-Vertrag überein. Das ändert dokumentiertes Verhalten (und `ServerMetricsTest`), daher Entscheidung des Nutzers. `getWindowUsage()` ist nicht betroffen: für das Fenster sendet upstream keinen Cache-Zähler.
- **Reproduktion:** ein Test mit `n_prompt_tokens_processed_total=10`, `n_prompt_tokens_cached_total=4` erwartet `getPromptTokens()==14` und `getCachedTokens()==4` – heute 10 und 0. Nicht geschrieben, solange die Richtung offen ist.
- **Rot-Grün-Nachweis:** keiner (kein Fix ohne Entscheidung).
- **Status:** nur beschrieben
- **Commit:** —

<a id="f59"></a>
### F59 – java-llama.cpp: `FfmNativeServer.isRunning()` kann nach einem parallelen `close()` freigegebenen Speicher lesen
- **Zeit:** 2026-10-17 21:25 UTC
- **Datei:** `llama-ffm/src/main/java/net/ladenthin/llama/ffm/FfmNativeServer.java` (`isRunning`, `close`)
- **Schwere:** niedrig
- **Beschreibung:** `isRunning()` prüft `closed` und ruft dann `jllama_server_is_running(handle)`. `close()` setzt `closed` und ruft `jllama_server_stop`, das laut `jllama_api.h` den Handle freigibt. Läuft `close()` (etwa aus einem Shutdown-Hook) zwischen der Prüfung und dem nativen Aufruf eines anderen Threads, liest `jllama_server_is_running` einen freigegebenen `jllama_server` – undefiniertes Verhalten im nativen Code, schlimmstenfalls ein JVM-Absturz. Der JNI-Zwilling `server.NativeServer` hält genau deshalb `isRunning()` und `close()` unter einem gemeinsamen `lifecycleLock`; die FFM-Klasse dokumentiert keine Einschränkung. (`FfmLlamaModel` hat dasselbe Muster, schreibt aber ausdrücklich, dass `close()` keinen neuen Aufruf überholen darf – dort ein dokumentierter Vertrag, kein Fund.)
- **Fix (Vorschlag, nicht umgesetzt):** ein privates Lock-Objekt um den Rumpf von `isRunning()` und `close()`, wie in `NativeServer`; `close()` blockiert ohnehin bis zum Ende des Workers.
- **Reproduktion:** nicht geschrieben – braucht einen Fake hinter den Method-Handles oder eine echte Bibliothek mit JDK 22+.
- **Rot-Grün-Nachweis:** keiner; in dieser Sandbox gibt es kein JDK 22+ (siehe Erkenntnisse).
- **CI-Bedarf:** Test im Job `test-java-llama-ffm` (z. B. paketinterne Naht für die beiden Method-Handles, `isRunning` im Latch festhalten, `close()` parallel, Erwartung: `close()` wartet).
- **Status:** nur beschrieben
- **Commit:** —

<a id="f60"></a>
### F60 – java-llama.cpp: Textblöcke einer Anthropic-Nachricht werden ohne Trenner zusammengeklebt
- **Zeit:** 2026-10-17 22:05 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/AnthropicApiSupport.java:145` (`appendOpenAiMessages`, Fall `text`)
- **Schwere:** niedrig
- **Beschreibung:** `/v1/messages` übersetzt eine Nachricht mit mehreren `text`-Blöcken in einen OpenAI-String, hängte die Blöcke aber ohne Trenner aneinander. Anthropic-Clients teilen einen Zug regelmäßig in mehrere Blöcke (Hinweisblock, dann der eigentliche Text); an der Nahtstelle verschmolzen Wörter (`first` + `second` → `firstsecond`). llama.cpps eigenes `/v1/messages` behält die Blöcke als Content-Parts und verbindet sie beim Rendern mit einem Zeilenumbruch (`concat_content_parts` in `common/chat.cpp`, b11639). Der Bildpfad derselben Methode reicht bereits Parts weiter und war nicht betroffen; die System-Blöcke verbindet upstream ebenfalls ohne Trenner, sie bleiben unverändert.
- **Fix:** Vor jedem weiteren Textblock ein `\n`, wie upstream.
- **Reproduktion:** `AnthropicApiSupportTest#requestJoinsTheTextBlocksOfOneMessageWithANewline`
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: is "first\nsecond" but: was "firstsecond"“, 14 Tests, 1 Fehlschlag), mit Fix grün 14/14; Gegenprobe per `git stash` rot, nach `stash pop` grün. `AnthropicStreamTranslatorTest` 4/4, `OpenAiCompatServerHttpTest` 44/44, `LlamaArchitectureTest` 13/13, `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `76a3370`
- **Präzisierung (2026-10-17, 22:45 UTC):** Upstream verbindet Text-Parts nur dann mit `\n`, wenn die Chat-Vorlage ausschließlich String-Inhalt annimmt (`messages_inp_normalizer`, Zweig `only_string`, in `common/chat.cpp`); eine Vorlage mit typisiertem Inhalt bekommt die Parts als Array und rendert sie selbst, unter Umständen ohne Trenner. Die Übersetzung reicht aber keine Parts weiter, sondern einen fertigen String – sie muss sich also für einen Trenner entscheiden, und `\n` ist der, den upstream wählt, wenn es selbst verbindet. Der Fix bleibt damit sinnvoll; die frühere Aussage „upstream verbindet mit Zeilenumbruch“ galt nur für String-Vorlagen. Alternative für den Nutzer: Text-Blöcke wie Bilder als Content-Parts weiterreichen und die Entscheidung der Vorlage überlassen.

<a id="f61"></a>
### F61 – java-llama.cpp: Textteile eines Responses-Eingabeelements werden ohne Trenner zusammengeklebt
- **Zeit:** 2026-10-17 22:15 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/ResponsesApiSupport.java:183` (`inputContentText`)
- **Schwere:** niedrig
- **Beschreibung:** Musterdurchsicht nach F60. `/v1/responses` verband die `input_text`-Teile eines Eingabeelements ohne Trenner, sodass an der Naht Wörter verschmolzen (`first` + `second` → `firstsecond`). Upstreams `server_chat_convert_responses_to_chatcmpl` (b11639, `server-chat.cpp`) behält die Teile als Content-Parts, die `concat_content_parts` in `common/chat.cpp` mit einem Zeilenumbruch verbindet. Der Bildpfad (`inputContentWithImages`) reicht bereits Parts weiter und war nicht betroffen.
- **Fix:** Vor jedem weiteren Textteil ein `\n`, wie upstream und wie F60.
- **Reproduktion:** `ResponsesApiSupportTest#requestJoinsTheTextPartsOfOneInputItemWithANewline`
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: is "first\nsecond" but: was "firstsecond"“, 11 Tests, 1 Fehlschlag), mit Fix grün 11/11; Gegenprobe per `git stash` rot, nach `stash pop` grün. `ResponsesStreamTranslatorTest` 4/4, `OpenAiCompatServerHttpTest` 44/44, `LlamaArchitectureTest` 13/13, `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `745fed5`
- **Präzisierung (2026-10-17, 22:45 UTC):** Upstream verbindet Text-Parts nur dann mit `\n`, wenn die Chat-Vorlage ausschließlich String-Inhalt annimmt (`messages_inp_normalizer`, Zweig `only_string`, in `common/chat.cpp`); eine Vorlage mit typisiertem Inhalt bekommt die Parts als Array und rendert sie selbst, unter Umständen ohne Trenner. Die Übersetzung reicht aber keine Parts weiter, sondern einen fertigen String – sie muss sich also für einen Trenner entscheiden, und `\n` ist der, den upstream wählt, wenn es selbst verbindet. Der Fix bleibt damit sinnvoll; die frühere Aussage „upstream verbindet mit Zeilenumbruch“ galt nur für String-Vorlagen. Alternative für den Nutzer: Text-Blöcke wie Bilder als Content-Parts weiterreichen und die Entscheidung der Vorlage überlassen.

<a id="f62"></a>
### F62 – java-llama.cpp: LangChain4j-Adapter klebt mehrere `TextContent` ohne Trenner zusammen
- **Zeit:** 2026-10-17 22:20 UTC
- **Datei:** `llama-langchain4j/src/main/java/net/ladenthin/llama/langchain4j/LangChain4jMapping.java:261` (`toJllamaUserMessage`)
- **Schwere:** niedrig
- **Beschreibung:** Musterdurchsicht nach F60/F61. Eine `UserMessage` aus mehreren `TextContent` ohne Medien wird zu einem String ohne Trenner (`"Hello"` + `"world"` → `"Helloworld"`). Enthält dieselbe Nachricht zusätzlich ein Bild, bleiben die Teile Content-Parts, die llama.cpp mit `\n` verbindet (`concat_content_parts`) – ein Bild ändert also, wie die Texte zusammenkommen. Anders als bei F60/F61 gibt es kein upstream-Gegenstück als Maßstab, und der Test `LangChain4jMappingTest#flattensMultiTextUserMessageToText` schreibt das heutige Verhalten mit `"Hello "` (Leerzeichen am Ende) fest.
- **Fix (Vorschlag, nicht umgesetzt):** wie F60/F61 mit `\n` verbinden (oder die Teile als Parts weiterreichen); dabei den festschreibenden Test anpassen. Entscheidung des Nutzers, weil der Test das Verhalten bewusst zu wählen scheint.
- **Reproduktion:** `UserMessage.from(TextContent.from("Hello"), TextContent.from("world"))` → `getContent()` = `"Helloworld"`.
- **Rot-Grün-Nachweis:** keiner (kein Fix ohne Entscheidung).
- **Status:** nur beschrieben
- **Commit:** —

<a id="f63"></a>
### F63 – java-llama.cpp: Responses-API-Objekte tragen `created_at: 0`
- **Zeit:** 2026-10-17 22:25 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/ResponsesApiSupport.java:286` (`newResponseShell`)
- **Schwere:** niedrig
- **Beschreibung:** `created_at` ist in der Responses-API und in llama.cpps eigenem `/v1/responses` (`to_json_oaicompat_resp`, b11639: `{"created_at", t}`) die Unix-Zeit der Erstellung. Die Schale, aus der jede Antwort und jedes Stream-Ereignis von `/v1/responses` entsteht, setzte die Konstante `0`; Clients zeigten jede Antwort als am 1. Januar 1970 erstellt. Die übrigen Oberflächen des Servers (`LlamaModelBackend`, Ollama) verwenden die aktuelle Zeit.
- **Fix:** `System.currentTimeMillis() / 1000L`.
- **Reproduktion:** `ResponsesApiSupportTest#responseCarriesTheCurrentUnixTimeAsCreatedAt`
- **Rot-Grün-Nachweis:** ohne Fix rot („a value equal to or greater than <1792275470L> <0L> was less than <1792275470L>“, 12 Tests, 1 Fehlschlag), mit Fix grün 12/12; Gegenprobe per `git stash` rot, nach `stash pop` grün. `ResponsesStreamTranslatorTest` 4/4, `OpenAiCompatServerHttpTest` 44/44, `LlamaArchitectureTest` 13/13, `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `aba327e`

<a id="f64"></a>
### F64 – java-llama.cpp: Ollama-Werkzeugaufrufe im Verlauf ohne `type` – llama.cpp lehnt die Anfrage ab
- **Zeit:** 2026-10-17 22:35 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OllamaApiSupport.java:186` (`toOpenAiMessage`)
- **Schwere:** mittel
- **Beschreibung:** Ollama schickt die Werkzeugaufrufe einer Assistenten-Runde als `{"function": {"name", "arguments"}}` ohne `type`. Die Übersetzung nach OpenAI wandelte nur die Argumente in einen String und ließ `type` weg. llama.cpps OpenAI-Nachrichtenparser verlangt ihn aber für jeden Werkzeugaufruf (`throw std::invalid_argument("Missing tool call type: …")` in `common/chat.cpp`, b11639). Jede `/api/chat`-Anfrage, die einen früheren Werkzeugaufruf im Verlauf mitschickt – also jede zweite Runde eines Gesprächs mit Werkzeugen –, scheiterte deshalb mit 400, bevor sie das Modell erreichte. Der bestehende Test zur Argumentumwandlung prüfte genau diese Form, aber nur die Argumente.
- **Fix:** Fehlt `type`, wird `"function"` gesetzt (der einzige Typ, den upstream annimmt).
- **Reproduktion:** `OllamaApiSupportTest#toOpenAiChatRequestAddsTheFunctionTypeToAssistantToolCalls`
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: is "function" but: was """, 22 Tests, 1 Fehlschlag), mit Fix grün 22/22; Gegenprobe per `git stash` rot, nach `stash pop` grün. `OpenAiCompatServerHttpTest` 44/44, `LlamaArchitectureTest` 13/13, `spotbugs:check` grün.
- **CI-Bedarf:** der Ende-zu-Ende-Beweis (die 400 von upstream) braucht ein Modell; der modellgebundene `OpenAiCompatServerIntegrationTest` lief hier übersprungen.
- **Status:** gefixt
- **Commit:** java-llama.cpp `f4c3ed6`

<a id="f65"></a>
### F65 – java-llama.cpp: CORS-Preflight erlaubt nur `Content-Type, Authorization` – Browser-SDKs werden abgewiesen
- **Zeit:** 2026-10-17 23:10 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiCompatServer.java` (`buildCorsFilter`)
- **Schwere:** niedrig
- **Beschreibung:** Bei gesetztem `corsAllowOrigin` beantwortete der Filter jeden Preflight mit `Access-Control-Allow-Headers: Content-Type, Authorization`. Der Server liest aber selbst `x-api-key` (F25) und bedient `/v1/messages`, dessen Browser-Clients `x-api-key`, `anthropic-version` und die `x-stainless-*`-Header der offiziellen SDKs schicken. Fehlt ein angefragter Header in der Antwort, verwirft der Browser die eigentliche Anfrage – ein im Browser laufender Anthropic-Client erreichte den Server also nie, obwohl CORS eingeschaltet war.
- **Fix:** Der Filter spiegelt die `Access-Control-Request-Headers` des Preflights (der Origin bleibt das Tor); ohne angefragte Header gilt `Content-Type, Authorization, x-api-key, anthropic-version`.
- **Reproduktion:** `OpenAiCompatServerHttpTest#optionsPreflightAllowsTheRequestedHeaders` (roher Socket, `OPTIONS /v1/messages` mit `content-type, x-api-key, anthropic-version, x-stainless-os`)
- **Rot-Grün-Nachweis:** ohne Fix rot (Antwort `access-control-allow-headers: content-type, authorization`, 45 Tests, 1 Fehlschlag), mit Fix grün 45/45; Gegenprobe per `git stash` rot, nach `stash pop` grün. `OllamaApiSupportTest` 22/22, `LlamaArchitectureTest` 13/13, `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `f782c7a`

<a id="f67"></a>
### F67 – java-llama.cpp: `setLoraAdapters` verspricht ein Leeren der KV-Caches, das upstream auf diesem Weg nicht macht
- **Zeit:** 2026-10-17 23:40 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/LlamaModel.java:1124` (Javadoc `setLoraAdapters`), ebenso `README.md:833`
- **Schwere:** niedrig (Dokumentation; das Verhalten selbst ist upstreams)
- **Beschreibung:** Javadoc und README sagen: „The native side clears affected KV caches when the effective adapter set changes.“ `setLoraAdapters` geht über `jllama_set_lora_adapters` → `engine::set_lora_adapters` → `SERVER_TASK_TYPE_SET_LORA`; upstream (`server-context.cpp`, b11639) setzt dort nur `params_base.lora_adapters = construct_lora_list(...)`. Eine spätere Anfrage ohne eigene `lora`-Angabe übernimmt in `launch_slot_with_task` im `else`-Zweig `slot.lora = params_base.lora_adapters`, **ohne** `slot.prompt.clear()` – den Cache leert upstream nur im Zweig für LoRA pro Anfrage (`lora_should_clear_cache`). Eine Anfrage, deren Prompt-Präfix noch im Slot liegt, rechnet also mit KV-Einträgen weiter, die unter den alten Adapter-Skalen entstanden sind, und liefert stillschweigend eine Mischung aus altem und neuem Adapter. Wer sich auf die Javadoc verlässt, rechnet damit nicht.
- **Fix:** keiner (Nutzerregel: kein Fix ohne roten Test). Möglich wäre, die Doku zu korrigieren (Hinweis auf `eraseSlot`/`cache_prompt=false` nach einem Wechsel) oder in `engine::set_lora_adapters` nach dem Task die Slots zu leeren (C++, außerhalb des Ziels).
- **Reproduktion:** keine lauffähige: es braucht ein Modell und einen dazu passenden LoRA-Adapter (nicht in `models.csv`), und in dieser Sandbox gibt es keine native Bibliothek und kein `gguf-py`. Ein Test würde denselben Prompt vor und nach `setLoraAdapter(0, 0f)` mit Greedy-Sampling schicken und mit einem frisch geladenen Modell ohne Adapter vergleichen.
- **CI-Bedarf:** ein kleines Modell mit LoRA-Adapter im Modellsatz (oder ein zur Laufzeit erzeugtes Zufallsmodell samt Adapter, wie `test_c_api.cpp` es für das Modell macht).
- **Status:** nur beschrieben
- **Commit:** –

<a id="f68"></a>
### F68 – java-llama.cpp: `RouterClient.awaitModelLoaded` läuft bei einem schlafenden Modell in den Timeout
- **Zeit:** 2026-10-17 23:55 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/RouterClient.java` (`awaitModelLoaded`)
- **Schwere:** niedrig
- **Beschreibung:** Mit `--sleep-idle-seconds` schläft ein geladener Worker im Leerlauf und wacht bei der nächsten Anfrage auf; `GET /models` meldet ihn dann als `sleeping`. Upstreams eigenes Warten auf ein Modell (`server_models`, `server-models.cpp:1501`/`1550`, b11639) behandelt `sleeping` ausdrücklich wie `loaded` („child is sleeping but still running; new request will wake it up“). `awaitModelLoaded` nahm nur `LOADED` an: wer ein bereits geladenes, aber leerlaufendes Modell abwartete, bekam nach der vollen Wartezeit `IllegalStateException` („did not reach LOADED … last status: sleeping“), obwohl das Modell bereit war.
- **Fix:** `SLEEPING` zählt wie `LOADED` als bereit; Javadoc und README-Kommentar angepasst.
- **Reproduktion:** `RouterClientTest#awaitModelLoaded_sleepingModelIsReady` (Stub-Server liefert `sleeping`)
- **Rot-Grün-Nachweis:** ohne Fix rot (`IllegalStateException: Model 'qwen' did not reach LOADED within 2000 ms (last status: qwen [sleeping])`), mit Fix grün, `RouterClientTest` 19/19; Gegenprobe per `git stash` rot, nach `stash pop` grün. `LlamaArchitectureTest` 13/13, `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `352057b`

<a id="f69"></a>
### F69 – java-llama.cpp: `withNPredict` dokumentiert `-2 = until context filled`, der Server lehnt `-2` ab
- **Zeit:** 2026-10-18 00:05 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/parameters/InferenceParameters.java:197` (Javadoc `withNPredict`)
- **Schwere:** niedrig (Dokumentation; der Aufrufer bekommt eine Ausnahme, kein stilles Fehlverhalten)
- **Beschreibung:** Die Javadoc nennt für den Anfragewert `n_predict` „-1 = infinity, -2 = until context filled“. Upstream (`tools/server/server-schema.cpp:44`, b11639) deklariert das Feld mit `set_hard_limits(-1, INT32_MAX)`, und `is_hard_limit` heißt laut `server-schema.h:52` „throw error if the value is invalid“: `withNPredict(-2)` endet also in einer 400 bzw. `InvalidRequestException`, statt bis zum Kontextende zu generieren. `-2` gibt es nur für die Startoption `-n` (`ModelParameters.setNPredict`, dort korrekt dokumentiert), nicht pro Anfrage.
- **Fix:** keiner (Nutzerregel: kein Fix ohne roten Test; der Test braucht ein geladenes Modell). Möglich: Javadoc auf „-1 = unbegrenzt, 0 = nur Prompt auswerten“ (upstreams `set_desc`) korrigieren oder `-2` in Java mit klarer Meldung abweisen.
- **Reproduktion:** keine lauffähige hier (kein Modell, keine native Bibliothek). Ein Test in einer modellgestützten Klasse: `model.complete(InferenceParameters.empty().withPrompt("x").withNPredict(-2))` → `InvalidRequestException` mit `n_predict` in der Meldung.
- **CI-Bedarf:** ein modellgestützter Java-Test (jedes Modell aus `models.csv` genügt).
- **Status:** nur beschrieben
- **Commit:** –

<a id="f70"></a>
### F70 – java-llama.cpp: `OpenAiServerCli` bewirbt mit `--parallel N` das ganze `-c` statt des Kontexts je Slot
- **Zeit:** 2026-10-18 00:15 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiServerCli.java` (`Options.toServerConfig`)
- **Schwere:** mittel
- **Beschreibung:** Aus `-c` leitet die CLI die beworbenen Budgets ab: `maxInputTokens`/`maxOutputTokens` (die Copilot & Co. zum Zuschneiden des Prompts nutzen) und daraus die Kontextlänge in `/api/show` und `/props`. Bei expliziter `--parallel N` vereinheitlicht llama.cpp den KV-Cache nicht (`-kvu`: „default: enabled if number of slots is auto“, `common/arg.cpp`, b11639), jeder Slot und damit jede Anfrage bekommt `n_ctx / N`, und upstreams eigenes `/props` meldet genau diese Slot-Größe (`slot_n_ctx`, `server-context.cpp:5083`). Mit `-c 8192 --parallel 4` warb der Server aber 8192 Token: Clients füllten Prompts auf das Vierfache dessen, was eine Anfrage fassen kann, und liefen in „exceeds the available context size“.
- **Fix:** Bei `parallel > 1` wird das Fenster je Slot (`ctx / parallel`) als Grundlage der Budgets genommen; ohne `--parallel` (Slotzahl „auto“, vereinheitlichter Cache) bleibt es das ganze `-c`.
- **Reproduktion:** `OpenAiServerCliTest#parallelSlotsShareTheContextSoTheBudgetIsPerSlot`
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: is <2048> but: was <8192>“), mit Fix grün, `OpenAiServerCliTest` 39/39; Gegenprobe per `git stash` rot, nach `stash pop` grün. `OpenAiCompatServerHttpTest` 45/45, `LlamaArchitectureTest` 13/13, `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `c1eaa50`

<a id="f71"></a>
### F71 – java-llama.cpp: Feintuning scheitert an jedem Zeichen außerhalb der BMP (Emoji) im Trainingstext
- **Zeit:** 2026-10-18 00:35 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/parameters/TrainingParameters.java` (`toJson`), Gegenstück `llama/src/main/cpp/train_engine.cpp:141`
- **Schwere:** niedrig
- **Beschreibung:** `LlamaTrainer.finetune` übergibt `TrainingParameters.toJson()`, und die JNI-Schicht liest den String mit `GetStringUTFChars` – Modified UTF-8. Jackson schreibt Nicht-ASCII roh in die JSON, ein Zeichen außerhalb der BMP (Emoji, seltene CJK-Zeichen) kommt deshalb als zwei 3-Byte-Surrogate (CESU-8) beim Parser an. Das vendorte nlohmann am Pin lehnt das ab – gemessen mit `vendor/nlohmann/json.hpp` von b11639: echtes UTF-8 `F0 9F 98 80` wird gelesen, die CESU-8-Form `ED A0 BD ED B8 80` endet in `parse_error.101 … invalid string: ill-formed UTF-8 byte`. Der ganze Lauf scheiterte also mit „invalid training config“, sobald Trainingstext oder ein Pfad ein solches Zeichen enthielt. Die beiden anderen JNI-Stellen, die Text übernehmen (`jllama.cpp:248`, `native_server.cpp:33`), umgehen `GetStringUTFChars` ausdrücklich; nur der Trainer nicht.
- **Fix:** `toJson` maskiert jedes Zeichen > 0x7F als `\uXXXX` (`escapeNonAscii`): ASCII ist in beiden Kodierungen gleich, und nlohmann setzt die Maskierungen samt Surrogatpaaren wieder zu richtigem UTF-8 zusammen. Bewusst nicht über `ESCAPE_NON_ASCII` am Mapper: `ObjectNode.toString()` nutzt Jacksons eigenen Mapper, und `writeValueAsString` hätte eine geprüfte Ausnahme eingeführt (SpotBugs `EXS_EXCEPTION_SOFTENING_NO_CONSTRAINTS`).
- **Reproduktion:** `TrainingParametersTest#nonAsciiTextIsEscapedSoTheJsonCrossesJniIntact` (nur ASCII in der JSON, Rücklesen ergibt den Originaltext); der native Fehler selbst ist mit dem nlohmann-Probelauf oben belegt.
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: is <0L> but: was <6L>“ Nicht-ASCII-Zeichen), mit Fix grün, `TrainingParametersTest` 8/8; Gegenprobe per `git stash` rot, nach `stash pop` grün. `LlamaArchitectureTest` 13/13, `spotbugs:check` grün.
- **CI-Bedarf:** ein Ende-zu-Ende-Beweis bräuchte `LlamaTrainerIntegrationTest` mit einem Emoji im Trainingstext (Modell `stories260K.gguf` liegt in CI vor).
- **Status:** gefixt
- **Commit:** java-llama.cpp `ca6ccdf`

<a id="f72"></a>
### F72 – java-llama.cpp: `GgufInspector` liest verschachtelte Arrays rekursiv – `StackOverflowError` statt `IOException`
- **Zeit:** 2026-10-18 01:05 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/GgufInspector.java` (`readArray`)
- **Schwere:** niedrig
- **Beschreibung:** `readArray` liest den Elementtyp und ruft für jedes Element `readValue` – auch für den Typ `ARRAY`, also rekursiv und ohne Tiefengrenze. Upstreams Leser lehnt ein Array mit Elementtyp `ARRAY` dagegen ab (`ggml/src/gguf.cpp:606`, b11639: „key … has invalid GGUF type“), llama.cpp kann eine solche Datei also nicht laden. Folgen: (1) eine präparierte Datei mit tiefer Verschachtelung (12 Byte und ein Stapelrahmen je Ebene) beendet `GgufInspector.read` mit `StackOverflowError`, obwohl die Methode für eine ungültige Datei `IOException` deklariert – wer den Inspektor vor dem Laden als Prüfung nutzt (`catch (IOException)`), fängt das nicht; (2) schon zwei Ebenen meldet der Inspektor als gültige Metadaten.
- **Fix:** `readArray` wirft bei Elementtyp `ARRAY` eine `IOException` mit dem Typ in der Meldung, wie upstream.
- **Reproduktion:** `GgufInspectorTest#nestedArrayIsRejectedLikeUpstream` (2 Ebenen) und `#deeplyNestedArraysFailWithIOExceptionNotStackOverflow` (100 000 Ebenen)
- **Rot-Grün-Nachweis:** ohne Fix rot („expected: <java.io.IOException> but was: <java.lang.StackOverflowError>“ bzw. „Expected java.io.IOException to be thrown, but nothing was thrown“), mit Fix grün, `GgufInspectorTest` 14 (1 übersprungen: der bestehende Test mit echtem Modell); Gegenprobe per `git stash` rot, nach `stash pop` grün. `LlamaArchitectureTest` 13/13, `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `79f3ee9`

<a id="f74"></a>
### F74 – java-llama.cpp: Ollama-`/api/generate` mit `suffix` (FIM) verwirft `stop`, `top_p`, `top_k` und `seed`
- **Zeit:** 2026-10-18 01:50 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OllamaApiSupport.java:403` (`toInfillRequest`)
- **Schwere:** mittel
- **Beschreibung:** Ein `/api/generate` ohne `suffix` übernimmt aus `options` `temperature`, `top_p`, `top_k`, `seed`, `num_predict` und `stop`. Mit `suffix`, also als Fill-in-the-Middle-Anfrage, übernahm `toInfillRequest` nur `temperature` und `num_predict`. Gerade FIM-Clients (Autovervollständigung über den Ollama-Provider) begrenzen ihre Vervollständigung mit `options.stop` (etwa `"\n\n"`); ohne sie lief jede Vervollständigung bis `num_predict` weiter. Upstreams `/infill` (`server-context.cpp`, `post_infill`, b11639) gibt den ganzen Body an `handle_completions_impl` und damit an denselben Schema-Parser wie `/completions`, liest also `stop`, `top_p`, `top_k` und `seed` genauso.
- **Fix:** `toInfillRequest` übernimmt dieselben Optionen wie `toOpenAiCompletionRequest` (`num_predict` weiter als `n_predict`, dem Namen des nativen Infill-Bodys).
- **Reproduktion:** `OllamaApiSupportTest#generateWithSuffixKeepsTheSamplingOptionsOfAPlainGenerate`
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: is "\n\n" but: was ""“), mit Fix grün (23/23 der Klasse, skipped 0); Gegenprobe per `git stash` rot, nach `pop` grün. `OpenAiCompatServerHttpTest` 45/45, `LlamaArchitectureTest` 13/13, `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `367592f`

<a id="f75"></a>
### F75 – java-llama.cpp: Responses-API verwirft `reasoning.effort` – `"none"` schaltet das Denken nicht ab
- **Zeit:** 2026-10-18 02:00 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/ResponsesApiSupport.java:51` (`toOpenAiChatRequest`)
- **Schwere:** mittel
- **Beschreibung:** `toOpenAiChatRequest` baut einen neuen Chat-Request und übernimmt nur Nachrichten, Werkzeuge, `temperature`, `top_p` und `max_output_tokens`. Das Feld `reasoning: {"effort": …}` des Responses-Protokolls fiel weg. upstreams eigener Responses-Umsetzer (`server-chat.cpp`, `convert_responses_to_chatcmpl`, b11639) bildet es auf `reasoning_effort` ab, und der Chat-Parser (`server-common.cpp:1420`) liest es: `"none"` schaltet das Denken der Vorlage ab, ein anderer Wert geht als `reasoning_effort` an die Vorlage. Ein Responses-Client, der `effort: "none"` (oder `"low"`) schickt, bekam bei uns trotzdem den vollen Denkblock samt der Token, die er kostet.
- **Fix:** ein textuelles `reasoning.effort` wird als `reasoning_effort` übernommen, wie upstream.
- **Reproduktion:** `ResponsesApiSupportTest#requestMapsTheReasoningEffortLikeUpstream`
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: is "none" but: was ""“), mit Fix grün (13/13 der Klasse, skipped 0); Gegenprobe per `git stash` rot, nach `pop` grün. `OpenAiCompatServerHttpTest` 45/45, `LlamaArchitectureTest` 13/13, `spotbugs:check` grün.
- **Nachtrag (2026-10-18):** wirkt erst zusammen mit [F77](#f77): der Chat-Mapper, durch den jede übersetzte Anfrage läuft, verwarf das erzeugte Feld bis dahin.
- **Status:** gefixt
- **Commit:** java-llama.cpp `485f153`

<a id="f76"></a>
### F76 – java-llama.cpp: Anthropic-`/v1/messages` verwirft `thinking.budget_tokens` – das Modell denkt unbegrenzt
- **Zeit:** 2026-10-18 02:05 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/AnthropicApiSupport.java:108` (`toOpenAiChatRequest`)
- **Schwere:** mittel
- **Beschreibung:** Ein Anthropic-Client begrenzt das Denken mit `thinking: {"type": "enabled", "budget_tokens": N}`. `toOpenAiChatRequest` übernahm `max_tokens`, `temperature`, `top_p`, `top_k` und `stop_sequences`, aber nicht `thinking`. upstreams `convert_anthropic_to_oai` (`server-chat.cpp:600`, b11639) setzt daraus `thinking_budget_tokens` (Vorgabe 10000 ohne `budget_tokens`), das der Chat-Parser (`server-common.cpp:1441`) als `reasoning_budget_tokens` an den Sampler gibt, der den Denkblock nach N Token schließt. Bei uns galt das Budget des Servers (Vorgabe unbegrenzt): ein Denkblock konnte die ganze Ausgabe aufbrauchen.
- **Fix:** ein `thinking` mit `type: "enabled"` wird zu `thinking_budget_tokens` (`budget_tokens` oder upstreams Vorgabe 10000). Andere Typen (`disabled`) bleiben wie bei upstream ohne Wirkung.
- **Reproduktion:** `AnthropicApiSupportTest#requestMapsAnEnabledThinkingBudgetLikeUpstream`
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: is <1024> but: was <0>“), mit Fix grün (15/15 der Klasse, skipped 0); Gegenprobe per `git stash` rot, nach `pop` grün. `AnthropicStreamTranslatorTest` 4/4, `OpenAiCompatServerHttpTest` 45/45, `LlamaArchitectureTest` 13/13, `spotbugs:check` grün.
- **Nicht übernommen:** upstreams Vorgabe `max_tokens` 4096, wenn es fehlt (bei uns unbegrenzt; das Anthropic-Protokoll verlangt das Feld), und das Durchreichen von `chat_template_kwargs` – beide bleiben als offener Verdacht stehen.
- **Nachtrag (2026-10-18):** wirkt erst zusammen mit [F77](#f77): der Chat-Mapper, durch den jede übersetzte Anfrage läuft, verwarf das erzeugte Feld bis dahin.
- **Status:** gefixt
- **Commit:** java-llama.cpp `5117413`

<a id="f77"></a>
### F77 – java-llama.cpp: `OpenAiRequestMapper` verwirft `chat_template_kwargs`, `reasoning_effort` und das Reasoning-Budget jedes Chat-Requests
- **Zeit:** 2026-10-18 02:20 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiRequestMapper.java:222` (`applyReasoningControls`, neu; Aufruf in `toInferenceParameters`)
- **Schwere:** mittel
- **Beschreibung:** Jede Chat-Anfrage an `OpenAiCompatServer` – OpenAI direkt sowie Ollama, Anthropic und Responses nach ihrer Übersetzung – läuft durch `OpenAiRequestMapper.toInferenceParameters`, das `InferenceParameters` aus einer festen Feldliste baut. Die Reasoning-Steuerungen, die upstreams Chat-Parser liest (`oaicompat_chat_params_parse`, `server-common.cpp:1403`–`1453`, b11639), fehlten darin: `chat_template_kwargs` (etwa `{"enable_thinking": false}`, mit dem Qwen3-Clients das Denken abschalten), `reasoning_effort` (`"none"` → Denken aus, sonst Template-Variable) und `reasoning_budget_tokens`/`thinking_budget_tokens`. Alle drei wurden still verworfen; `InferenceParameters` hat die passenden Setter (`withChatTemplateKwargs`, `withReasoningBudgetTokens`). Folge: die Fixes F75 (Responses `reasoning.effort`) und F76 (Anthropic `thinking.budget_tokens`) erzeugten das richtige Feld, das danach verloren ging.
- **Fix:** `applyReasoningControls` übernimmt die Template-Variablen des Requests, setzt bei `reasoning_effort: "none"` `enable_thinking` auf `false` (und entfernt `reasoning_effort`), legt jeden anderen Wert als Template-Variable `reasoning_effort` ab und übernimmt das Budget (`reasoning_budget_tokens`, sonst `thinking_budget_tokens`) – die Regeln von upstreams Parser. Class-Javadoc nachgezogen.
- **Reproduktion:** `OpenAiRequestMapperTest#chatTemplateKwargsAreForwarded`, `#reasoningEffortNoneSwitchesThinkingOffLikeUpstream`, `#reasoningEffortLevelBecomesATemplateVariableLikeUpstream`, `#reasoningBudgetIsForwardedUnderEitherUpstreamName`, `#anthropicAndResponsesReasoningControlsReachTheInferenceParameters` (die Kette Übersetzer → Mapper)
- **Rot-Grün-Nachweis:** ohne Fix alle fünf rot (u. a. „Expected: is <512> but: was <0>“, „Expected: is "low" but: was ""“, „Expected: is <1024> but: was <0>“), mit Fix 21/21 der Klasse grün, skipped 0; Gegenprobe per `git stash` wieder 5 rot, nach `pop` grün. `AnthropicApiSupportTest` 15/15, `ResponsesApiSupportTest` 13/13, `OllamaApiSupportTest` 23/23, `OpenAiCompatServerHttpTest` 45/45, `LlamaArchitectureTest` 13/13, `spotbugs:check` grün.
- **CI-Bedarf:** ein modellgestützter Lauf (`OpenAiCompatServerIntegrationTest` mit dem Reasoning-Modell) würde zeigen, dass `reasoning_effort: "none"` den Denkblock wirklich unterdrückt; hier gibt es kein Modell.
- **Status:** gefixt
- **Commit:** java-llama.cpp `60c2b9f`

<a id="f78"></a>
### F78 – java-llama.cpp: Ollama-`/api/chat` ignoriert `think` – `"think": false` schaltet das Denken nicht ab
- **Zeit:** 2026-10-18 02:30 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OllamaApiSupport.java:143` (`toOpenAiChatRequest`)
- **Schwere:** mittel
- **Beschreibung:** Ollamas Chat-API steuert das Denken eines Modells mit dem Top-Level-Feld `think`: `false` schaltet es ab, `true` an, ein String wählt bei gpt-oss eine Stufe (`"low"`, `"medium"`, `"high"`). `OllamaApiSupport.toOpenAiChatRequest` las das Feld nicht. Ein Ollama-Client, der für schnelle Antworten `think: false` schickt (Open WebUI, ollama-python), bekam trotzdem einen vollen Denkblock samt seiner Token und Latenz. Gefunden in der Musterdurchsicht nach F77.
- **Fix:** `think: false` wird zu `reasoning_effort: "none"`, ein nicht leerer String zu `reasoning_effort: <Stufe>`; `true` bleibt bei der Vorgabe der Vorlage. Der Chat-Mapper wendet `reasoning_effort` seit F77 nach upstreams Regeln an (`enable_thinking: false` bzw. Template-Variable).
- **Reproduktion:** `OllamaApiSupportTest#toOpenAiChatRequestHonoursThinkLikeOllama` (Kette Übersetzer → `OpenAiRequestMapper` → `InferenceParameters.toJson()`)
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: is <false> but: was <true>“ – `enable_thinking` fehlte), mit Fix grün (24/24 der Klasse, skipped 0); Gegenprobe per `git stash` rot, nach `pop` grün. `OpenAiRequestMapperTest` 21/21, `OpenAiCompatServerHttpTest` 45/45, `LlamaArchitectureTest` 13/13, `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `d252f84`

<a id="f79"></a>
### F79 – java-llama.cpp: gestreamtes `/v1/completions` ignoriert `suffix`, `best_of` und `echo`, die der nicht gestreamte Weg abweist
- **Zeit:** 2026-10-18 02:40 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiRequestMapper.java:89` (`toCompletionParameters`)
- **Schwere:** niedrig
- **Beschreibung:** Nicht gestreamt geht `/v1/completions` unverändert an `handleCompletionsOai`, also an upstreams `oaicompat_completion_params_parse` (`server-common.cpp:1082`, b11639), das `echo: true` („Only no echo is supported“) sowie `suffix` und `best_of` („Unsupported param: …“) mit 400 abweist. Gestreamt baut `toCompletionParameters` die Anfrage aus einer Teilmenge und überging diese Felder still. Ein gestreamter FIM-Request mit `suffix` lieferte so eine bloße Fortsetzung des Präfixes, die über den Suffix hinausläuft und ihn im Editor dupliziert, statt eines Fehlers; dieselbe Anfrage verhielt sich gestreamt und nicht gestreamt verschieden.
- **Fix:** `toCompletionParameters` weist `echo: true`, `suffix` und `best_of` mit upstreams Meldungen ab (`IllegalArgumentException`, vom Server als Anfragefehler gemeldet).
- **Reproduktion:** `OpenAiRequestMapperTest#completionParamsUpstreamRejectsAreRejectedOnTheStreamedPathToo`
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected java.lang.IllegalArgumentException to be thrown, but nothing was thrown.“), mit Fix grün (22/22 der Klasse, skipped 0); Gegenprobe per `git stash` rot, nach `pop` grün. `OpenAiCompatServerHttpTest` 45/45, die drei Übersetzer-Tests, `LlamaArchitectureTest` 13/13, `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `7545586`

<a id="f80"></a>
### F80 – java-llama.cpp: Anthropic-Antwort verwirft den Denkinhalt – kein `thinking`-Block
- **Zeit:** 2026-10-18 02:50 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/AnthropicApiSupport.java:297` (`toAnthropicResponse`)
- **Schwere:** niedrig
- **Beschreibung:** Die native OpenAI-Antwort trägt den Denkinhalt eines Reasoning-Modells in `message.reasoning_content`. `toAnthropicResponse` übernahm nur `content` und `tool_calls`, der Denkinhalt ging verloren. upstreams `to_json_anthropic` (`server-task.cpp:738`, b11639) setzt ihn als ersten Inhaltsblock `{"type": "thinking", "thinking": …, "signature": ""}`. Ein Anthropic-Client, der das Denken eingeschaltet hat (seit F76 auch wirksam), bekam es nie zu sehen; ein Client, der die Gedanken in der nächsten Runde zurückschickt, konnte das nicht.
- **Fix:** ein nicht leeres `reasoning_content` wird wie upstream als erster Block `thinking` mit leerer Signatur ausgegeben.
- **Reproduktion:** `AnthropicApiSupportTest#responseCarriesTheReasoningAsAThinkingBlockFirstLikeUpstream`
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: is <2> but: was <1>“), mit Fix grün (16/16 der Klasse, skipped 0); Gegenprobe per `git stash` rot, nach `pop` grün. `AnthropicStreamTranslatorTest` 4/4, `OpenAiCompatServerHttpTest` 45/45, `LlamaArchitectureTest` 13/13, `spotbugs:check` grün.
- **Nicht im Fix:** der gestreamte Pfad (`AnthropicStreamTranslator`, upstream `thinking_delta`) und die Responses-Antwort (upstream ein `reasoning`-Item mit `reasoning_text`) – offener Verdacht.
- **Status:** gefixt
- **Commit:** java-llama.cpp `e709e5f`

<a id="f81"></a>
### F81 – java-llama.cpp: gestreamte Anthropic-Antwort verwirft den Denkinhalt – keine `thinking_delta`s
- **Zeit:** 2026-10-18 03:00 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/AnthropicStreamTranslator.java:80` (`onChunk`, `end`), Ereignisse in `AnthropicApiSupport`
- **Schwere:** niedrig
- **Beschreibung:** Das Gegenstück zu F80 im Stream. Die nativen OpenAI-Chunks tragen den Denkinhalt als `delta.reasoning_content`; `AnthropicStreamTranslator` übersetzte nur `delta.content` und Werkzeugaufrufe. upstreams `to_json_anthropic_stream` (`server-task.cpp:1356`, `:907`, b11639) öffnet dafür einen Block `{"type": "thinking", "thinking": ""}`, schickt `thinking_delta`s und schließt ihn mit einem leeren `signature_delta`, das das Protokoll vor dem Schließen eines thinking-Blocks verlangt.
- **Fix:** Der Übersetzer öffnet beim ersten `reasoning_content` einen thinking-Block, schickt die Teile als `thinking_delta` und schließt ihn mit `signature_delta` + `content_block_stop`, sobald Text beginnt oder der Stream endet; die Blöcke folgen einander, statt wie bei upstream bis zum Ende offen zu bleiben. Drei neue Ereignis-Bausteine in `AnthropicApiSupport`.
- **Reproduktion:** `AnthropicStreamTranslatorTest#reasoningDeltasStreamAsAThinkingBlockClosedWithASignatureBeforeTheText`
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: a string containing "\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}" but: was ""“), mit Fix grün (5/5 der Klasse, skipped 0); Gegenprobe per `git stash` beider Dateien rot, nach `pop` grün. `AnthropicApiSupportTest` 16/16, `OpenAiCompatServerHttpTest` 45/45, `LlamaArchitectureTest` 13/13, `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `d2afd35`

<a id="f82"></a>
### F82 – java-llama.cpp: Responses-Antwort verwirft den Denkinhalt – kein `reasoning`-Item
- **Zeit:** 2026-10-18 03:10 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/ResponsesApiSupport.java:237` (`toResponsesResponse`)
- **Schwere:** niedrig
- **Beschreibung:** Gegenstück zu F80 für die Responses-API. `toResponsesResponse` übernahm aus der nativen OpenAI-Antwort `content` und `tool_calls`; `message.reasoning_content` ging verloren. upstreams `to_json_oaicompat_resp` (`server-task.cpp:531`, b11639) gibt ihn als erstes `output`-Item aus: `{"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": …}], "encrypted_content": "", "status": "completed"}`. Ein Responses-Client sah die Gedanken nie, auch nicht, wenn er sie (seit F75/F77 wirksam) mit `reasoning.effort` angefordert hatte.
- **Fix:** ein nicht leeres `reasoning_content` wird wie upstream als erstes `reasoning`-Item ausgegeben (Id `rs_<responseId>`).
- **Reproduktion:** `ResponsesApiSupportTest#responseCarriesTheReasoningAsAReasoningItemFirstLikeUpstream`
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: is <2> but: was <1>“), mit Fix grün (14/14 der Klasse, skipped 0); Gegenprobe per `git stash` rot, nach `pop` grün. `ResponsesStreamTranslatorTest` 4/4, `OpenAiCompatServerHttpTest` 45/45, `LlamaArchitectureTest` 13/13, `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `070dbce`

<a id="f83"></a>
### F83 – java-llama.cpp: gestreamte Responses-Antwort verwirft den Denkinhalt – keine `response.reasoning_text.delta`s
- **Zeit:** 2026-10-18 03:20 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/ResponsesStreamTranslator.java:87` (`onChunk`, `end`)
- **Schwere:** niedrig
- **Beschreibung:** Gegenstück zu F82 im Stream. `ResponsesStreamTranslator` übersetzte `delta.content` und Werkzeugaufrufe, `delta.reasoning_content` ging verloren. upstreams `to_json_oaicompat_resp_stream` (`server-task.cpp:1198`, `:591`, b11639) öffnet ein `reasoning`-Item (`response.output_item.added`), streamt `response.reasoning_text.delta` und setzt das Item mit dem ganzen Text an den Anfang des abschließenden `output`.
- **Fix:** Beim ersten Denkteil öffnet der Übersetzer ein `reasoning`-Item (`rs_<responseId>`), streamt `response.reasoning_text.delta` mit `item_id`/`output_index`/`content_index` wie die Textdeltas, schließt mit `response.reasoning_text.done` und `response.output_item.done` und setzt das Item vorn in `response.completed.output`. Die beiden Delta-Ereignisse teilen sich einen Helfer (SpotBugs PRMC).
- **Reproduktion:** `ResponsesStreamTranslatorTest#reasoningDeltasStreamAsAReasoningItemLikeUpstream`
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: a string containing "event: response.output_item.added" but: was ""“), mit Fix grün (5/5 der Klasse, skipped 0); Gegenprobe per `git stash` rot, nach `pop` grün. `ResponsesApiSupportTest` 14/14, `OpenAiCompatServerHttpTest` 45/45, `LlamaArchitectureTest` 13/13, `spotbugs:check` grün (erst nach dem Helfer).
- **Status:** gefixt
- **Commit:** java-llama.cpp `17ad313`

<a id="f84"></a>
### F84 – java-llama.cpp: Anthropic-Verlauf verwirft `thinking`-Blöcke früherer Runden
- **Zeit:** 2026-10-18 03:30 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/AnthropicApiSupport.java:166` (`appendOpenAiMessages`)
- **Schwere:** niedrig
- **Beschreibung:** Ein Anthropic-Client schickt die `thinking`-Blöcke früherer Assistenten-Runden im Verlauf mit (Pflicht bei Werkzeugschleifen mit Denken). `appendOpenAiMessages` ignorierte den Blocktyp; eine Runde nur aus Denkinhalt wurde zu `content: ""` ohne Gedanken. upstreams `convert_anthropic_to_oai` (`server-chat.cpp:415`, `:524`–`536`, b11639) sammelt die Blöcke als `reasoning_content` der Nachricht, das `common_chat_msgs_parse_oaicompat` liest und Vorlagen, die frühere Gedanken rendern (etwa bei verschachteltem Denken mit Werkzeugen), in den Prompt setzen. Bei uns sah das Modell einen anderen Verlauf als bei upstream.
- **Fix:** `thinking`-Blöcke werden gesammelt und als `reasoning_content` an die Nachricht gehängt; eine Nachricht nur mit Denkinhalt wird wie upstream mit `content: ""` und `reasoning_content` übernommen. Der Chat-Mapper reicht `messages` unverändert weiter, das Feld erreicht also die native Seite.
- **Reproduktion:** `AnthropicApiSupportTest#requestKeepsThinkingBlocksOfTheHistoryAsReasoningContentLikeUpstream`
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: is "6 times 7" but: was ""“), mit Fix grün (17/17 der Klasse, skipped 0); Gegenprobe per `git stash` rot, nach `pop` grün. `AnthropicStreamTranslatorTest` 5/5, `OpenAiRequestMapperTest` 22/22, `OpenAiCompatServerHttpTest` 45/45, `LlamaArchitectureTest` 13/13, `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `fb226b3`

<a id="f85"></a>
### F85 – java-llama.cpp: Responses-`reasoning`-Eingabe wird zur Benutzernachricht
- **Zeit:** 2026-10-18 03:40 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/ResponsesApiSupport.java:199` (`appendReasoningItem`, neu; Aufruf in `appendInput`)
- **Schwere:** mittel
- **Beschreibung:** Ein Responses-Client schickt die `reasoning`-Items früherer Antworten im `input` zurück (Codex & Co. tun das in jeder Werkzeugschleife). `appendInput` kannte den Typ nicht; er fiel in den `default`-Zweig für Nachrichten, und weil ein reasoning-Item keine `role` hat, wurde daraus `role: "user"` mit dem Text seiner Inhaltsteile. Der frühere Gedankengang des Modells stand damit als Benutzereingabe im Prompt – das Modell sah den Nutzer seine eigenen Überlegungen sagen. upstreams `convert_responses_to_chatcmpl` (`server-chat.cpp:226`–`255`, b11639) hängt den Text als `reasoning_content` an die vorherige Assistenten-Nachricht (`merge_prev`) oder legt eine neue Assistenten-Nachricht an. Gefunden als Gegenstück zu F84.
- **Fix:** `appendReasoningItem` übernimmt den Text von `content[0].text` als `reasoning_content` der letzten Assistenten-Nachricht, sonst einer neuen mit leerem `content`. Ein Item ohne lesbaren Text (nur `summary`/`encrypted_content`) wird übersprungen; upstream weist es mit 400 ab, hier wäre das für Clients mit verschlüsselten Gedanken eine unnötige Ablehnung.
- **Reproduktion:** `ResponsesApiSupportTest#reasoningInputItemsBecomeAssistantReasoningNotUserTextLikeUpstream` (neue Assistenten-Nachricht nach einem Nutzer, Zusammenführung mit einem `function_call`, verschlüsseltes Item übersprungen)
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: is "q" but: was "plan"“ – der Denktext als Benutzernachricht), mit Fix grün (15/15 der Klasse, skipped 0); Gegenprobe per `git stash` rot, nach `pop` grün. `ResponsesStreamTranslatorTest` 5/5, `OpenAiRequestMapperTest` 22/22, `OpenAiCompatServerHttpTest` 45/45, `LlamaArchitectureTest` 13/13, `spotbugs:check` grün (nach Herausziehen zweier doppelter Aufrufe, PRMC).
- **Status:** gefixt
- **Commit:** java-llama.cpp `7a8b262`

<a id="f86"></a>
### F86 – java-llama.cpp: Responses-Werkzeugergebnis als Teil-Array kommt leer beim Modell an
- **Zeit:** 2026-10-18 03:50 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/ResponsesApiSupport.java:203` (`toolOutputParts`, neu; Aufruf in `appendInput`)
- **Schwere:** mittel
- **Beschreibung:** Ein `function_call_output` darf sein `output` als String oder als Array aus `input_text`-/`input_image`-Teilen schicken (z. B. ein Werkzeug, das einen Screenshot liefert). `appendInput` las `output.asText("")`; für ein Array ist das `""`, das Werkzeugergebnis erreichte das Modell also leer und es arbeitete ohne Ergebnis weiter. upstreams `convert_responses_to_chatcmpl` (`server-chat.cpp:189`–`220`, b11639) macht aus den Teilen `text`- und `image_url`-Teile der `role: tool`-Nachricht.
- **Fix:** ein Array-`output` wird zu Inhaltsteilen (`input_text` → `text`, `input_image` → `image_url`), andere Teiltypen werden übersprungen (upstream weist sie ab); ein String bleibt wie bisher.
- **Reproduktion:** `ResponsesApiSupportTest#functionCallOutputArrayKeepsItsPartsLikeUpstream`
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: is "text" but: was ""“; die erste Fassung des Tests lief über `get(0)` in eine NPE und wurde auf `path(0)` umgestellt, damit der Fehler an der Assertion sichtbar wird), mit Fix grün (16/16 der Klasse, skipped 0); Gegenprobe per `git stash` rot, nach `pop` grün. `ResponsesStreamTranslatorTest` 5/5, `OpenAiRequestMapperTest` 22/22, `OpenAiCompatServerHttpTest` 45/45, `LlamaArchitectureTest` 13/13, `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `28054ba`

<a id="f87"></a>
### F87 – java-llama.cpp: Responses-`function_call`-Items werden einzelne Assistenten-Runden statt einer Runde mit mehreren Aufrufen
- **Zeit:** 2026-10-18 04:00 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/ResponsesApiSupport.java:198` (`assistantToolCalls`, neu; Aufruf in `appendInput`)
- **Schwere:** niedrig
- **Beschreibung:** Die Responses-API gibt parallele Werkzeugaufrufe als aufeinanderfolgende `function_call`-Items zurück, oft nach einem Text des Assistenten. `appendInput` legte für jedes Item eine eigene Assistenten-Nachricht an: aus „Text, Aufruf c1, Aufruf c2, Ergebnis c1, Ergebnis c2“ wurden drei Assistenten-Runden, und die Ergebnisse zu c1 folgten erst nach der Runde von c2. upstreams `convert_responses_to_chatcmpl` (`server-chat.cpp:177`–`187`, b11639) hängt jeden Aufruf an die vorherige Assistenten-Nachricht (`merge_prev`); die Vorlage sieht eine Runde mit zwei Aufrufen, wie sie das Modell erzeugt hat.
- **Fix:** ein `function_call` hängt sich an die `tool_calls` der letzten Assistenten-Nachricht (das Array wird bei Bedarf angelegt), sonst entsteht eine neue Assistenten-Nachricht mit `content: null`. Folge wie bei upstream: ein `reasoning`-Item nach einem Aufruf ohne Ergebnis dazwischen überschreibt den Denktext derselben Nachricht (`prev_msg["reasoning_content"] = …`).
- **Reproduktion:** `ResponsesApiSupportTest#consecutiveFunctionCallsJoinThePrecedingAssistantMessageLikeUpstream`; der F85-Test wurde auf eine realistische Werkzeugschleife umgestellt (Denktext, Aufruf, Ergebnis, Denktext, Antwort), weil seine erste Folge die getrennten Nachrichten der alten Variante voraussetzte.
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: is <4> but: was <6>“, ebenso der umgestellte F85-Test), mit Fix grün (17/17 der Klasse, skipped 0); Gegenprobe per `git stash` rot (2 Fehlschläge), nach `pop` grün. `ResponsesStreamTranslatorTest` 5/5, `OpenAiRequestMapperTest` 22/22, `OpenAiCompatServerHttpTest` 45/45, `LlamaArchitectureTest` 13/13, `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `77a6903`

<a id="f88"></a>
### F88 – java-llama.cpp: Anthropic-`tool_result` verwirft Bilder
- **Zeit:** 2026-10-18 04:10 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/AnthropicApiSupport.java:257` (`toolResultContent`, neu; Aufruf in `appendOpenAiMessages`)
- **Schwere:** mittel
- **Beschreibung:** Werkzeuge liefern ihr Ergebnis als `tool_result` mit Inhaltsblöcken, darunter Bilder (ein Screenshot, eine Bilddatei, die ein Agent wie Claude Code liest). `toolResultText` übernahm nur die Textblöcke; das Bild erreichte das Modell nie, ein Vision-Modell antwortete, als hätte das Werkzeug keins geliefert. upstreams `convert_anthropic_to_oai` (`server-chat.cpp:453`–`512`, b11639) macht aus einem Ergebnis mit Bildern Inhaltsteile `text`/`image_url` der `role: tool`-Nachricht und lässt nur ein reines Text-Ergebnis als String. Das Anthropic-Gegenstück zu F86, gefunden in der Durchsicht der Typ-Weichen gegen upstreams Liste.
- **Fix:** `toolResultContent` gibt bei einem Bild die Teile in Blockreihenfolge zurück (`text` und `image_url` über das vorhandene `imageUrl`), sonst wie bisher den Text.
- **Reproduktion:** `AnthropicApiSupportTest#toolResultImagesReachTheToolMessageLikeUpstream`
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: is "cat.png" but: was ""“ – der Inhalt war ein String), mit Fix grün (18/18 der Klasse, skipped 0); Gegenprobe per `git stash` rot, nach `pop` grün. `AnthropicStreamTranslatorTest` 5/5, `OpenAiRequestMapperTest` 22/22, `OpenAiCompatServerHttpTest` 45/45, `LlamaArchitectureTest` 13/13, `spotbugs:check` grün (nach `new TextNode(...)` statt `textNode(...)`, AI_ANNOTATION_ISSUES_NEEDS_NULLABLE).
- **Status:** gefixt
- **Commit:** java-llama.cpp `ba93540`

<a id="f89"></a>
### F89 – java-llama.cpp: Responses-`input_file` wird still verworfen statt abgewiesen
- **Zeit:** 2026-10-18 04:20 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/ResponsesApiSupport.java:240` (`rejectInputFiles`, neu), `llama/src/main/java/net/ladenthin/llama/server/OpenAiCompatServer.java:703` (`handleResponses`)
- **Schwere:** niedrig
- **Beschreibung:** Hängt ein Responses-Client eine Datei als Inhaltsteil `input_file` an, übernahmen `inputContentText`/`inputContentWithImages` nur Text- und Bildteile; die Datei fiel weg und das Modell antwortete, als wäre sie nie geschickt worden (eine „Zusammenfassung“ ohne Dokument). upstreams `convert_responses_to_chatcmpl` (`server-chat.cpp:92`, b11639) antwortet darauf mit 400 „'input_file' is not supported by llamacpp at this moment“. Beim Fix zeigte sich zudem: `handleResponses` rief den Übersetzer außerhalb der Fehlerbehandlung auf, eine dort geworfene `IllegalArgumentException` hätte den Handler ohne Antwort verlassen.
- **Fix:** `rejectInputFiles` wirft bei einem `input_file`-Teil eine `IllegalArgumentException` mit upstreams Meldung (plus Datei-Kennung, SpotBugs WEM); `handleResponses` meldet eine Ablehnung des Übersetzers als 400.
- **Reproduktion:** `OpenAiCompatServerHttpTest#responsesWithAnInputFileIsRejectedLikeUpstream`
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: is <400> but: was <200>“), mit Fix grün (46/46 der Klasse, skipped 0); Gegenprobe per `git stash` beider Dateien rot, nach `pop` grün. `ResponsesApiSupportTest` 17/17, `ResponsesStreamTranslatorTest` 5/5, `OpenAiRequestMapperTest` 22/22, `LlamaArchitectureTest` 13/13, `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `3280821`

<a id="f90"></a>
### F90 – java-llama.cpp: Claude Codes wechselnder `cch`-Stempel im System-Prompt verhindert jeden Prefix-Cache-Treffer
- **Zeit:** 2026-10-18 04:30 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/AnthropicApiSupport.java:144` (`normalizeBillingHeader`, neu; Aufruf in `systemText`)
- **Schwere:** mittel (Leistung)
- **Beschreibung:** Claude Code beginnt seinen System-Prompt mit `x-anthropic-billing-header: cc_version=…; cc_entrypoint=cli; cch=a5145;…`, und der `cch`-Wert ändert sich bei jeder Anfrage. Der Prompt unterscheidet sich damit schon in den ersten Token, und der KV-Präfix-Cache (`cache_prompt`, vom Mapper auf `true` gesetzt) trifft nie: jede Runde einer Agenten-Sitzung rechnete den ganzen, viele Tausend Token langen System-Prompt samt Verlauf neu. upstreams `server_chat_convert_anthropic_to_oai` ruft dafür `normalize_anthropic_billing_header` (`server-chat.cpp:326`, llama.cpp #21793, b11639) und überschreibt die fünf Stempelzeichen mit `fffff`.
- **Fix:** `normalizeBillingHeader` macht dasselbe wie upstream (nur bei diesem Präfix, nur wenn nach den fünf Zeichen ein `;` folgt), angewendet auf den String- und jeden Text-Block des `system`-Felds.
- **Reproduktion:** `AnthropicApiSupportTest#billingHeaderStampIsNormalizedSoThePrefixCacheHitsLikeUpstream` (Block- und String-Form, zwei verschiedene Stempel ergeben denselben Text)
- **Rot-Grün-Nachweis:** ohne Fix rot (Stempel `a5145` unverändert), mit Fix grün (19/19 der Klasse, skipped 0); Gegenprobe per `git stash` rot, nach `pop` grün. `AnthropicStreamTranslatorTest` 5/5, `OpenAiRequestMapperTest` 22/22, `OpenAiCompatServerHttpTest` 46/46, `LlamaArchitectureTest` 13/13, `spotbugs:check` grün.
- **CI-Bedarf:** dass der Cache danach tatsächlich trifft (`cached_tokens` > 0 in der zweiten Runde), zeigt nur ein modellgestützter Lauf.
- **Status:** gefixt
- **Commit:** java-llama.cpp `7d08a0c`

<a id="f91"></a>
### F91 – java-llama.cpp: Responses-Verlauf spaltet jede Assistenten-Runde in Denk- und Antwortnachricht
- **Zeit:** 2026-10-18 04:40 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/ResponsesApiSupport.java:257` (`joinContent`, `lastAssistant`, neu; Aufruf im `message`-Zweig von `appendInput`)
- **Schwere:** niedrig
- **Beschreibung:** Ein Responses-Client gibt jede frühere Runde als `reasoning`-Item und danach als `message`-Item mit `role: assistant` zurück. Seit F85 wird das reasoning-Item zu einer Assistenten-Nachricht mit `reasoning_content`; das folgende `message`-Item legte dann eine **zweite** Assistenten-Nachricht mit der Antwort an. Die Vorlage sah jede Runde als zwei aufeinanderfolgende Assistenten-Runden, eine ohne Antwort und eine ohne Denkinhalt. upstreams `convert_responses_to_chatcmpl` (`server-chat.cpp:150`, b11639) führt ein Assistenten-`message`-Item mit der vorherigen Assistenten-Nachricht zusammen (`merge_prev`), wie bei Werkzeugaufrufen (F87). Gefunden beim Lesen des upstream-Umsetzers von oben bis unten (Regel aus F90).
- **Fix:** Folgt ein Assistenten-`message`-Item auf eine Assistenten-Nachricht, wird sein Inhalt angehängt: ein leerer oder `null`-Inhalt ersetzt, zwei Texte mit Zeilenumbruch verbunden (wie `concat_content_parts`), sonst Teil-Arrays zusammengefügt.
- **Reproduktion:** `ResponsesApiSupportTest#anAssistantMessageJoinsThePrecedingAssistantTurnLikeUpstream`; der F85-Test erwartet seine abschließende Antwort jetzt in der Nachricht mit dem zweiten Denktext.
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: is <3> but: was <4>“, ebenso der angepasste F85-Test), mit Fix grün (18/18 der Klasse, skipped 0); Gegenprobe per `git stash` rot (2 Fehlschläge), nach `pop` grün. `ResponsesStreamTranslatorTest` 5/5, `OpenAiRequestMapperTest` 22/22, `OpenAiCompatServerHttpTest` 46/46, `LlamaArchitectureTest` 13/13, `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `04d4cdb`

### F92 – java-llama.cpp: blockierende LangChain4j-Antwort verliert den Denkinhalt
- **Zeit:** 2026-10-18 05:10 UTC
- **Datei:** `llama-langchain4j/src/main/java/net/ladenthin/llama/langchain4j/LangChain4jMapping.java:206` (`toAiMessage`)
- **Schwere:** niedrig
- **Beschreibung:** `JllamaChatModel` (blockierend) baute die `AiMessage` nur aus Text und Werkzeugaufrufen; der `reasoning_content` der nativen Antwort ging verloren, `AiMessage.thinking()` war immer `null`. Der Streaming-Pfad (`StreamingChunkAssembler`) setzt ihn, und die Modul-README sagt, Reasoning erscheine als `AiMessage.thinking()`. Ein Wechsel von `JllamaStreamingChatModel` zu `JllamaChatModel` ließ den Denkinhalt also stillschweigend verschwinden.
- **Fix:** `toAiMessage` liest den Denkinhalt mit `ChatResponseParser.extractChoiceReasoningContent` aus dem rohen JSON und setzt ihn in beiden Zweigen (mit und ohne Werkzeugaufrufe), wenn er nicht leer ist.
- **Reproduktion:** `LangChain4jMappingTest#blockingResponseCarriesTheThinkingLikeTheStreamedOne`
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: is "6 times 7"“), mit Fix grün (21/21 der Klasse, skipped 0); Gegenprobe per `git stash` rot, nach `pop` grün; alle Modultests grün (Integrationstests ohne Modell übersprungen). Das Modul deklariert weder Spotless noch SpotBugs (weder eigene noch Eltern-POM), diese Gates entfallen hier.
- **Status:** gefixt
- **Commit:** java-llama.cpp `6caa722`

### F93 – java-llama.cpp: gestreamte LangChain4j-Antwort hat nie eine Token-Nutzung
- **Zeit:** 2026-10-18 04:45 UTC
- **Datei:** `llama-langchain4j/src/main/java/net/ladenthin/llama/langchain4j/LangChain4jMapping.java:105` (`toStreamingParameters`)
- **Schwere:** niedrig
- **Beschreibung:** `StreamingChunkAssembler` übernimmt die Token-Nutzung aus dem abschließenden `usage`-Chunk. upstream (`server_task_result_cmpl_final::to_json_oaicompat_chat_stream`, `server-task.cpp:494`, b11639) sendet diesen Chunk aber nur bei `stream_options.include_usage`, und `toStreamingParameters` setzte das nie. Die gestreamte `ChatResponse` von `JllamaStreamingChatModel` hatte daher immer `tokenUsage() == null`, die blockierende dagegen die echte Zählung. Die Modul-README verspricht beides („both the blocking and the streamed final response carry the model's real finish reason and token usage“). Gefunden über den Vergleich blockierender und gestreamter Pfad (Erkenntnis aus F92).
- **Fix:** `toStreamingParameters` setzt `stream_options` `{"include_usage":true}` vor dem Customizer (der es weiterhin überschreiben kann). `ChatStreamChunkParser` reicht den Usage-Chunk des Endschritts bereits einzeln weiter, der Assembler liest ihn.
- **Reproduktion:** `LangChain4jMappingTest#streamingParametersRequestTheUsageChunk`
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: is <true> but: was <false>“), mit Fix grün (22/22 der Klasse, skipped 0); Gegenprobe per `git stash` rot (1 Fehlschlag), nach `pop` grün; alle Modultests grün (`StreamingChunkAssemblerTest` 7/7, `JsonSchemaElementSerializerTest` 8/8, Integrationstests ohne Modell übersprungen). Spotless/SpotBugs sind im Modul nicht deklariert (siehe F92).
- **CI-Bedarf:** Der Integrationsjob `test-java-llama-langchain4j-integration` läuft mit Modell; ein Ende-zu-Ende-Nachweis der Zählung im Stream wäre dort möglich, ist aber nicht Voraussetzung.
- **Status:** gefixt
- **Commit:** java-llama.cpp `cde7159`

### F94 – java-llama.cpp: Textteile einer LangChain4j-Nutzernachricht werden ohne Trenner zusammengeklebt
- **Zeit:** 2026-10-18 04:55 UTC
- **Datei:** `llama-langchain4j/src/main/java/net/ladenthin/llama/langchain4j/LangChain4jMapping.java:287` (`toJllamaUserMessage`)
- **Schwere:** niedrig
- **Beschreibung:** Musterdurchsicht nach F60/F61 im LangChain4j-Adapter. Eine `UserMessage` mit mehreren `TextContent`-Teilen und ohne Medien wurde zu einem String verbunden, aber ohne Trenner: `first` + `second` → `firstsecond`. LangChain4j-Werkzeuge (RAG-Kontext plus Frage, Vorlagen) teilen Nutzerzüge regelmäßig so auf. upstream verbindet Textteile für Vorlagen ohne typisierten Inhalt mit `\n` (`concat_content_parts`, `common/chat.cpp`, b11639); der Adapter reicht keine Parts weiter, sondern muss selbst einen Trenner wählen. Der bestehende Test `flattensMultiTextUserMessageToText` schrieb das Verhalten nur mit einem Leerzeichen am Ende des ersten Teils fest, das den Fehler verdeckte.
- **Fix:** Verbinden mit `StringJoiner("\n")`, wie bei F60 (Anthropic) und F61 (Responses). Der festschreibende Test erwartet jetzt `"Hello \nworld"`.
- **Reproduktion:** `LangChain4jMappingTest#joinsTextPartsWithANewlineLikeUpstream`, dazu `#flattensMultiTextUserMessageToText`
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: is "first\nsecond" but: was "firstsecond"“, ebenso der angepasste Test), mit Fix grün (23/23 der Klasse, skipped 0); Gegenprobe per `git stash` rot (2 Fehlschläge), nach `pop` grün; alle Modultests grün (Integrationstests ohne Modell übersprungen). Spotless/SpotBugs sind im Modul nicht deklariert (siehe F92).
- **Status:** gefixt
- **Commit:** java-llama.cpp `ee99d85`

### F95 – java-llama.cpp: gestreamtes `/v1/completions` und der Chat-Pfad verwerfen llama.cpps Sampling-Felder
- **Zeit:** 2026-10-18 05:30 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiRequestMapper.java:200` (`applyLlamaCppSampling`, neu; `readMaxTokens`)
- **Schwere:** mittel
- **Beschreibung:** Aus dem offenen Verdacht vom 2026-10-18 (gestreamtes `/v1/completions`). Das nicht gestreamte `/v1/completions` reicht den Körper an upstreams Schema durch (`server-schema.cpp`, b11639), das gestreamte baut `InferenceParameters` im `OpenAiRequestMapper` aus einer festen Feldliste. Alles außerhalb dieser Liste ging verloren: `min_p`, `typical_p`, `repeat_penalty`, `repeat_last_n`, `top_n_sigma`, `xtc_*`, `dynatemp_*`, `min_keep`, `n_keep`, `ignore_eos`, `grammar`/`json_schema`, und `n_predict` (upstreams Name, `max_tokens`/`max_completion_tokens` sind seine Aliase und verlieren gegen ihn). Dieselbe Anfrage erzeugte gestreamt also andere Ausgaben, eine Grammatik wirkte nur ohne `stream`. Derselbe Mapper bedient beide Chat-Pfade, die Felder fehlten dort ebenso. Alle Felder haben bereits einen Setter; anders als im Verdacht befürchtet, braucht dieser Teil keinen API-Umbau.
- **Fix:** `applyLlamaCppSampling` (aus `applyCommonFields`) übernimmt die Felder mit den vorhandenen Settern; `json_schema` nur ohne `grammar`, wie upstream. `readMaxTokens` liest zuerst `n_predict`. Weiter offen (Verdacht bleibt): `logit_bias`, `n_probs`/`logprobs` (im gestreamten Completion-Chunk ohnehin nicht ausgegeben) und `n`.
- **Reproduktion:** `OpenAiRequestMapperTest#llamaCppSamplingExtensionsReachTheStreamedCompletion`, `#nPredictWinsOverMaxTokensLikeUpstream`, `#jsonSchemaReachesTheStreamedCompletionUnlessAGrammarIsGiven`
- **Rot-Grün-Nachweis:** ohne Fix rot (3 Fehlschläge: „Expected: is a numeric value within <1.0E-4> of <0.05> but: <0.0>“, „Expected: is <7> but: was <50>“, „Expected: is "object" but: was ""“), mit Fix grün (25/25 der Klasse, skipped 0); Gegenprobe per `git stash` rot (3), nach `pop` grün. `OpenAiCompatServerHttpTest` 46/46, `AnthropicApiSupportTest` 19/19, `ResponsesApiSupportTest` 18/18, `OllamaApiSupportTest` 24/24, `LlamaArchitectureTest` 13/13, `spotless:apply` und `spotbugs:check` grün (eine erste Fassung mit allen Feldern in einer Methode meldete `CE_CLASS_ENVY`, eine Map-Tabelle `DMC_DUBIOUS_MAP_COLLECTION`; jetzt je Feld ein Aufruf über `withFloat`/`withInt`).
- **Status:** gefixt
- **Commit:** java-llama.cpp `8ce9ffc`

### F96 – java-llama.cpp: Anthropic-Anfrage ohne `max_tokens` läuft unbegrenzt, `chat_template_kwargs` geht verloren
- **Zeit:** 2026-10-18 05:45 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/AnthropicApiSupport.java:111` (`toOpenAiChatRequest`)
- **Schwere:** niedrig
- **Beschreibung:** Aus dem offenen Verdacht „Anthropic-Rest nach F76“. upstreams `convert_anthropic_to_oai` (`server-chat.cpp:585`, b11639) setzt `max_tokens` 4096, wenn die Anfrage keines trägt, und reicht `chat_template_kwargs` durch. `/v1/messages` übernahm `max_tokens` nur, wenn vorhanden, eine Anfrage ohne generierte also, bis der Kontext voll war; und `chat_template_kwargs` (etwa Qwen3s `enable_thinking`) wurde verworfen, obwohl der nachgeschaltete `OpenAiRequestMapper` es seit F77 liest.
- **Fix:** Ohne Zahl in `max_tokens` wird `DEFAULT_MAX_TOKENS` (4096) gesetzt; ein Objekt in `chat_template_kwargs` wird kopiert.
- **Reproduktion:** `AnthropicApiSupportTest#requestWithoutMaxTokensIsCappedLikeUpstream`, `#requestPassesChatTemplateKwargsLikeUpstream` (zunächst ein Sammeltest, nach der Regel „ein Test je Richtung“ aufgeteilt)
- **Rot-Grün-Nachweis:** gegen die Produktionsdatei vor `d29e084` rot, jeder Teil für sich (2 Fehlschläge: „Expected: is <4096> but: was <0>“, „Expected: is <true> but: was <false>“ für `chat_template_kwargs`), mit Fix grün (21/21 der Klasse, skipped 0). `OpenAiCompatServerHttpTest` 46/46, `AnthropicStreamTranslatorTest` 5/5, `OpenAiRequestMapperTest` 25/25, `ResponsesApiSupportTest` 18/18, `OllamaApiSupportTest` 24/24, `LlamaArchitectureTest` 13/13, `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `d29e084`

### F97 – java-llama.cpp: Ollama-`/api/chat` verliert den Denkinhalt in beide Richtungen
- **Zeit:** 2026-10-18 06:00 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OllamaApiSupport.java:267` (`toOllamaChatResponse`, `toOllamaContentLine`, `toOpenAiMessage`)
- **Schwere:** niedrig
- **Beschreibung:** Aus dem offenen Verdacht „Ollama-Antworten tragen den Denkinhalt nicht“. Ollama legt den Denkinhalt eines Modells in `message.thinking` ab, blockierend und je Stream-Zeile, und Clients senden ihn im Verlauf zurück. `OllamaApiSupport` übernahm aus OpenAIs Antwort nur `content` und `tool_calls`; im Stream lieferte `toOllamaContentLine` für einen Chunk ohne Text `null`, Denk-Deltas fielen also ganz weg. Seit F78 kann ein Client mit `think` das Denken steuern, sah es aber nie. Im Verlauf blieb `thinking` als unbekannter Schlüssel stehen, den llama.cpps Parser ignoriert (`common/chat.cpp:422` liest `reasoning_content`), wie bei F84 für Anthropic.
- **Fix:** Blockierend und im Stream wird `reasoning_content` zu `message.thinking` (eine Denk-Zeile trägt `content` leer); im Verlauf wird `thinking` zu `reasoning_content`.
- **Reproduktion:** `OllamaApiSupportTest#blockingChatAnswerCarriesTheThinkingLikeOllama`, `#streamedThinkingDeltaBecomesAThinkingLineLikeOllama`, `#historyThinkingBecomesReasoningContent` (zunächst ein Sammeltest, nach der Regel „ein Test je Richtung“ aufgeteilt)
- **Rot-Grün-Nachweis:** gegen die Produktionsdatei vor `9eab6e0` rot, jede Richtung für sich (3 Fehlschläge: „Expected: is "6 times 7" but: was ""“ blockierend und im Verlauf, „Expected: is "6 ti" but: was ""“ im Stream), mit Fix grün (27/27 der Klasse, skipped 0). `OpenAiCompatServerHttpTest` 46/46, `AnthropicApiSupportTest` 20/20, `ResponsesApiSupportTest` 18/18, `OpenAiRequestMapperTest` 25/25, `LlamaArchitectureTest` 13/13, `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `9eab6e0`

### F98 – java-llama.cpp: gestreamtes Ollama-`/api/chat` meldet keine Tokenzahlen
- **Zeit:** 2026-10-18 06:15 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiCompatServer.java:545` (`streamOllamaChat`), `llama/src/main/java/net/ladenthin/llama/server/OllamaApiSupport.java` (`toOllamaDoneLine`, `usageOf`)
- **Schwere:** niedrig
- **Beschreibung:** Aus dem Nachtrag zum offenen Verdacht „`done_reason` uneinheitlich“. Das blockierende `/api/chat` meldet `prompt_eval_count`/`eval_count`, die Abschlusszeile des Streams nicht. `streamOllamaChat` forderte den Usage-Chunk auch nicht an (anders als die Anthropic- und Responses-Routen mit `withUsageChunk`), upstream sendet ihn nur bei `stream_options.include_usage`. Clients, die Token/s aus `eval_count` der Abschlusszeile berechnen (Ollamas eigene CLI mit `--verbose`, Open WebUI), sahen nichts. Gefunden über den Vergleich blockierender und gestreamter Pfad (Erkenntnis aus F92).
- **Fix:** `streamOllamaChat` fordert den Usage-Chunk an (`withUsageChunk`), merkt sich ihn über `OllamaApiSupport.usageOf` und gibt ihn an `toOllamaDoneLine(model, accumulator, usage)` weiter, das die Zählung wie die blockierende Antwort einträgt. `done_reason` bleibt unverändert (Teil des Verdachts, Entscheidung des Nutzers).
- **Reproduktion:** `OpenAiCompatServerHttpTest#ollamaChatStreamingDoneLineCarriesTheTokenCountsLikeTheBlockingAnswer` (das `FakeBackend` sendet den Usage-Chunk wie upstream nur bei `include_usage`)
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: is <12> but: was <-1>“), mit Fix grün (47/47 der Klasse, skipped 0); Gegenprobe per `git stash` beider Produktionsdateien rot (1), nach `pop` grün. `OllamaApiSupportTest` 27/27, `AnthropicApiSupportTest` 21/21, `ResponsesApiSupportTest` 18/18, `OpenAiRequestMapperTest` 25/25, `AnthropicStreamTranslatorTest` 5/5, `LlamaArchitectureTest` 13/13, `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `0cd9f8a`

### F99 – java-llama.cpp: blockierende Responses-Antwort trägt ein leeres `message`-Item neben Werkzeugaufrufen
- **Zeit:** 2026-10-18 06:30 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/ResponsesApiSupport.java:378` (`toResponsesResponse`)
- **Schwere:** niedrig
- **Beschreibung:** Vergleich blockierender und gestreamter Pfad (Erkenntnis aus F92). `toResponsesResponse` legte immer ein `message`-Item an, auch mit leerem Text; bei einer Runde, die nur Werkzeuge aufruft, stand so ein leeres `output_text` vor den `function_call`-Items. upstreams `to_json_oaicompat_resp` (`server-task.cpp`, b11639) legt das Item nur bei nicht leerem Text an, und unser eigener Stream (`ResponsesStreamTranslator`) öffnet es erst mit dem ersten Text-Delta. Dieselbe Anfrage ergab also blockierend eine andere `output`-Liste als gestreamt; ein Client, der das leere Item zurückschickt, erzeugt im nächsten Zug eine leere Assistenten-Nachricht im Verlauf. Der Test `responseEmitsFunctionCallItemsForToolCalls` schrieb das leere Item fest („output[0] is the (empty) message“).
- **Fix:** Das `message`-Item wird nur bei nicht leerem Text angelegt; der festschreibende Test erwartet den `function_call` jetzt an Stelle 0.
- **Reproduktion:** `ResponsesApiSupportTest#toolCallOnlyResponseHasNoEmptyMessageItemLikeUpstreamAndTheStream`, dazu `#responseEmitsFunctionCallItemsForToolCalls`
- **Rot-Grün-Nachweis:** ohne Fix rot (2 Fehlschläge: „Expected: is <1> but: was <2>“, „Expected: is "function_call" but: was "message"“), mit Fix grün (19/19 der Klasse, skipped 0); Gegenprobe per `git stash` rot (2), nach `pop` grün. `ResponsesStreamTranslatorTest` 5/5, `OpenAiCompatServerHttpTest` 47/47, `AnthropicApiSupportTest` 21/21, `OllamaApiSupportTest` 27/27, `OpenAiRequestMapperTest` 25/25, `LlamaArchitectureTest` 13/13, `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `9d0a021`

### F102 – java-llama.cpp: Chat-Anfrage mit `id_slot: -1` wird mit 400 abgewiesen
- **Zeit:** 2026-10-18 07:20 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiRequestMapper.java:133` (`applyCommonFields`)
- **Schwere:** niedrig
- **Beschreibung:** upstream liest `id_slot` mit dem Standardwert `-1` („beliebiger Slot“, `server-context.cpp:4845`, b11639). Der Mapper reichte jeden ganzzahligen Wert an `InferenceParameters.withSlotId` weiter, das negative Werte mit `IllegalArgumentException` ablehnt; der Server macht daraus 400. Ein Client, der den llama.cpp-Standard ausdrücklich mitschickt, scheiterte so auf beiden Chat-Pfaden und beim gestreamten `/v1/completions`, während das nicht gestreamte `/v1/completions` (wörtlich an upstream durchgereicht) dieselbe Anfrage annahm.
- **Fix:** Ein negativer `id_slot` wird nicht gesetzt, das entspricht upstreams Standard. `withSlotId` bleibt streng (ein Java-Aufrufer pinnt bewusst einen Slot).
- **Reproduktion:** `OpenAiRequestMapperTest#slotMinusOneMeansAnySlotLikeUpstream`
- **Rot-Grün-Nachweis:** ohne Fix rot („Unexpected exception thrown: java.lang.IllegalArgumentException: slotId must be non-negative but was -1“, als Assertion über `assertDoesNotThrow`), mit Fix grün (26/26 der Klasse, skipped 0); Gegenprobe per `git stash` rot (1), nach `pop` grün. `OpenAiCompatServerHttpTest` 47/47, `AnthropicApiSupportTest` 21/21, `ResponsesApiSupportTest` 19/19, `OllamaApiSupportTest` 27/27, `LlamaArchitectureTest` 13/13, `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `ded2762`

### F103 – java-llama.cpp: `max_tokens: 0` generiert unbegrenzt statt nur den Cache zu füllen
- **Zeit:** 2026-10-18 07:35 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiRequestMapper.java:164` (`applyCommonFields`)
- **Schwere:** niedrig
- **Beschreibung:** Musterdurchsicht nach F102 (Werte, die der Mapper anders behandelt als upstream). upstreams Schema (`server-schema.cpp`, `n_predict` mit den Aliasen `max_completion_tokens`/`max_tokens`, Grenzen -1..INT32_MAX, b11639): „When 0, no tokens will be generated but the prompt is evaluated into the cache“. Der Mapper übernahm die Obergrenze nur bei `> 0`; ein Client, der mit `0` den Cache vorwärmt, bekam auf den gemappten Wegen (beide Chat-Pfade, gestreamtes `/v1/completions`, und damit Anthropic/Responses/Ollama) eine Generierung ohne Obergrenze bis zum Kontextende, das wörtlich durchgereichte `/v1/completions` dagegen keine.
- **Fix:** Die Obergrenze wird bei `>= 0` gesetzt; `-1` bzw. kein Feld lässt upstreams Standard (unbegrenzt).
- **Reproduktion:** `OpenAiRequestMapperTest#zeroTokenCapOnlyWarmsTheCacheLikeUpstream`
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: is <0> but: was <-1>“), mit Fix grün (27/27 der Klasse, skipped 0); Gegenprobe per `git stash` rot (1), nach `pop` grün. `OpenAiCompatServerHttpTest` 47/47, `AnthropicApiSupportTest` 21/21, `ResponsesApiSupportTest` 19/19, `OllamaApiSupportTest` 27/27, `LlamaArchitectureTest` 13/13, `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `bcbafbc`

### F104 – java-llama.cpp: Ollama-Routen verwerfen `repeat_penalty`, `min_p` und weitere Optionen; `num_predict: -2` wird abgewiesen
- **Zeit:** 2026-10-18 07:50 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OllamaApiSupport.java:246` (`copyLlamaCppOptions`, `copyNumPredict`, neu; Aufrufe in `toOpenAiChatRequest`, `toOpenAiCompletionRequest`, `toInfillRequest`)
- **Schwere:** niedrig
- **Beschreibung:** Musterdurchsicht nach F102/F103 (Sonderwerte) und F74 (Optionsmengen der Geschwister). Ollama legt die Sampling-Optionen unter `options` mit llama.cpps Namen ab (`repeat_penalty`, `repeat_last_n`, `min_p`, `typical_p`, `presence_penalty`, `frequency_penalty`, `num_keep`), Clients wie Open WebUI setzen sie. Alle drei Ollama-Routen übernahmen nur `temperature`, `top_p`, `top_k`, `seed`, `num_predict` und `stop`; der Rest ging still verloren, obwohl jede Route ihn an upstreams Schema weitergeben kann (`/api/chat` seit F95 über den Mapper, `/api/generate` und FIM wörtlich). Dazu ein Sonderwert: Ollamas `num_predict: -2` („Kontext füllen“) ging unverändert als `max_tokens`/`n_predict` -2 weiter, upstreams harte Grenze beginnt bei -1 – `/api/generate` und FIM antworteten mit einem Fehler.
- **Fix:** Ein gemeinsamer Helfer kopiert die llama.cpp-benannten Optionen auf allen drei Routen (`num_keep` → `n_keep`); `repeat_last_n: -1` (Ollama: Kontextgröße, upstream: ungültig) bleibt beim Server-Standard; `num_predict` unter -1 wird -1.
- **Reproduktion:** `OllamaApiSupportTest#chatCopiesTheLlamaCppSamplingOptions`, `#generateCopiesTheLlamaCppSamplingOptions`, `#infillCopiesTheLlamaCppSamplingOptions`, `#numPredictMinusTwoFillsTheContextAsUpstreamsMinusOne`; `#repeatLastNMinusOneIsLeftToTheServerDefault` schützt die Ausnahme (ohne Fix grün, weil gar nichts kopiert wurde)
- **Rot-Grün-Nachweis:** ohne Fix rot (4 Fehlschläge: dreimal „Expected: is <0.05> but: was <0.0>“, einmal „Expected: is <-1> but: was <-2>“), mit Fix grün (32/32 der Klasse, skipped 0); Gegenprobe per `git stash` rot (4), nach `pop` grün. `OpenAiCompatServerHttpTest` 47/47, `AnthropicApiSupportTest` 21/21, `ResponsesApiSupportTest` 19/19, `OpenAiRequestMapperTest` 27/27, `LlamaArchitectureTest` 13/13, `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `0eba89d`

### F105 – java-llama.cpp: Responses-Anfrage mit `previous_response_id` wird ohne Verlauf beantwortet
- **Zeit:** 2026-10-18 08:05 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/ResponsesApiSupport.java:56` (`toOpenAiChatRequest`)
- **Schwere:** mittel
- **Beschreibung:** Gefunden beim Lesen von upstreams `server_chat_convert_responses_to_chatcmpl` (`server-chat.cpp`, b11639) von oben (Regel aus F90): Gleich nach der `input`-Prüfung lehnt upstream `previous_response_id` mit „llama.cpp does not support 'previous_response_id'.“ ab. `/v1/responses` hier ignorierte das Feld. Es gibt keinen Antwortspeicher, also beantwortete der Server einen Zug, der nur über die ID an den vorigen anknüpft (OpenAIs Standardweg mit `store`), ohne dessen Verlauf – still eine Antwort auf eine andere Frage.
- **Fix:** Ein nicht leeres `previous_response_id` wirft eine `IllegalArgumentException` (seit F89 eine 400) mit upstreams Wortlaut, dem Wert und dem Hinweis, die früheren Züge in `input` zu senden.
- **Reproduktion:** `ResponsesApiSupportTest#previousResponseIdIsRejectedLikeUpstream`
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected java.lang.IllegalArgumentException to be thrown, but nothing was thrown.“), mit Fix grün (20/20 der Klasse, skipped 0); Gegenprobe per `git stash` rot (1), nach `pop` grün. `ResponsesStreamTranslatorTest` 5/5, `OpenAiCompatServerHttpTest` 47/47, `AnthropicApiSupportTest` 21/21, `OllamaApiSupportTest` 32/32, `OpenAiRequestMapperTest` 27/27, `LlamaArchitectureTest` 13/13, `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `b9bfb45`

### F106 – java-llama.cpp: Responses-Anfrage verliert `top_k`, `seed`, `stop` und llama.cpps Erweiterungen
- **Zeit:** 2026-10-18 08:20 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/ResponsesApiSupport.java:146` (`toOpenAiChatRequest`, `CHAT_PASSTHROUGH_FIELDS` neu)
- **Schwere:** niedrig
- **Beschreibung:** Aus dem offenen Verdacht der vorigen Runde. upstreams `convert_responses_to_chatcmpl` (`server-chat.cpp`, b11639) beginnt mit `chatcmpl_body = response_body` und kopiert damit jedes Feld in die Chat-Anfrage, die der Chat-Parser liest. `ResponsesApiSupport` übernahm nur `temperature`, `top_p`, `max_output_tokens` und `reasoning.effort`; `top_k`, `seed`, `stop`, die Strafen, `chat_template_kwargs`, `response_format`, die Cache-Felder und llama.cpps Sampling-Erweiterungen gingen still verloren, obwohl der nachgeschaltete Mapper sie seit F95 liest (Muster F74/F104).
- **Fix:** Eine Liste der Felder, die der Chat-Mapper liest, wird unverändert übernommen; `input`, `tools` und `tool_choice` bleiben übersetzt statt kopiert.
- **Reproduktion:** `ResponsesApiSupportTest#requestFieldsTheChatMapperReadsPassThroughLikeUpstream` (parametrisiert, ein Fall je Feld, 27 Fälle)
- **Rot-Grün-Nachweis:** ohne Fix rot, jeder Fall für sich (27 Fehlschläge, z. B. „Expected: is <0.05> but: was <>“), mit Fix grün (47/47 der Klasse, skipped 0); Gegenprobe per `git stash` rot (27), nach `pop` grün. `ResponsesStreamTranslatorTest` 5/5, `OpenAiCompatServerHttpTest` 47/47, `AnthropicApiSupportTest` 21/21, `OllamaApiSupportTest` 32/32, `OpenAiRequestMapperTest` 27/27, `LlamaArchitectureTest` 13/13, `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `87635b2`

### F107 – java-llama.cpp: Anthropic-`tool_choice` `none` läuft als `auto`
- **Zeit:** 2026-10-18 08:30 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/AnthropicApiSupport.java:348` (`toOpenAiToolChoice`)
- **Schwere:** niedrig
- **Beschreibung:** Anthropics `tool_choice` `{"type":"none"}` verbietet Werkzeugaufrufe (Clients setzen es, um eine Runde ohne Werkzeug zu erzwingen, etwa für eine Zusammenfassung, während die Werkzeugliste für den Prompt-Cache gleich bleibt). `toOpenAiToolChoice` kannte nur `auto`, `any` und `tool`; `none` ergab kein `tool_choice`, die Anfrage lief als `auto`, und das Modell durfte die mitgeschickten Werkzeuge aufrufen. upstreams `convert_anthropic_to_oai` (b11639) hat dieselbe Lücke; der Chat-Mapper und llama.cpps Chat-Parser verstehen OpenAIs `tool_choice` `none` aber, die Übersetzung kostet eine Zeile. Hier also bewusst über upstream hinaus.
- **Fix:** `none` wird zu `tool_choice` `none`.
- **Reproduktion:** `AnthropicApiSupportTest#toolChoiceNoneForbidsToolCalls`
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: is "none" but: was ""“), mit Fix grün (22/22 der Klasse, skipped 0); Gegenprobe per `git stash` rot (1), nach `pop` grün. `AnthropicStreamTranslatorTest` 5/5, `OpenAiCompatServerHttpTest` 47/47, `ResponsesApiSupportTest` 47/47, `OllamaApiSupportTest` 32/32, `OpenAiRequestMapperTest` 27/27, `LlamaArchitectureTest` 13/13, `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `b85eed8`

### F108 – java-llama.cpp: Responses-`text.format` (strukturierte Ausgabe) wird ignoriert
- **Zeit:** 2026-10-18 08:45 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/ResponsesApiSupport.java:177` (`appendResponseFormat`, neu; Aufruf in `toOpenAiChatRequest`)
- **Schwere:** niedrig
- **Beschreibung:** Die Responses-API trägt strukturierte Ausgabe in `text.format` (`{"type":"json_schema","name","schema","strict"}` bzw. `{"type":"json_object"}`), die Chat-API in `response_format` mit dem Schema unter `json_schema`. `/v1/responses` übersetzte das nicht; ein Client, der ein Schema verlangte, bekam freien Text ohne Grammatik. upstreams `convert_responses_to_chatcmpl` (b11639) kopiert `text` unverändert in die Chat-Anfrage, wo der Chat-Parser es nicht liest – dieselbe Lücke. Der Chat-Mapper reicht `response_format` an die native Seite weiter, die Übersetzung reicht also.
- **Fix:** `json_schema` wird zu `response_format` mit `name`, `description`, `schema` und `strict` unter `json_schema`; `json_object` wird übernommen; `text` braucht nichts. Ein vom Client zusätzlich gesetztes `response_format` (seit F106 durchgereicht) hat Vorrang.
- **Reproduktion:** `ResponsesApiSupportTest#textFormatJsonSchemaBecomesTheChatResponseFormat`, `#textFormatJsonObjectBecomesTheChatResponseFormat`; `#textFormatTextLeavesTheResponseFormatOut` schützt den `text`-Fall (ohne Fix grün)
- **Rot-Grün-Nachweis:** ohne Fix rot (2 Fehlschläge: „Expected: is "json_schema" but: was ""“, „Expected: is "json_object" but: was ""“), mit Fix grün (50/50 der Klasse, skipped 0); Gegenprobe per `git stash` rot (2), nach `pop` grün. `ResponsesStreamTranslatorTest` 5/5, `OpenAiCompatServerHttpTest` 47/47, `AnthropicApiSupportTest` 22/22, `OllamaApiSupportTest` 32/32, `OpenAiRequestMapperTest` 27/27, `LlamaArchitectureTest` 13/13, `spotless:apply` und `spotbugs:check` grün (eine erste Fassung mit einem Array-Literal meldete `SACM_STATIC_ARRAY_CREATED_IN_METHOD`, jetzt eine Konstante).
- **Status:** gefixt
- **Commit:** java-llama.cpp `941357f`

### F109 – java-llama.cpp: `tool_choice`, das eine Funktion benennt, läuft als `auto`
- **Zeit:** 2026-10-18 09:00 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiRequestMapper.java:74`, `llama/src/main/java/net/ladenthin/llama/server/ResponsesApiSupport.java` (`toOpenAiChatRequest`, Werkzeugblock)
- **Schwere:** niedrig
- **Beschreibung:** Musterdurchsicht nach F107. OpenAI-Chat benennt ein Werkzeug mit `tool_choice` `{"type":"function","function":{"name":"f"}}`, die Responses-API mit `{"type":"function","name":"f"}`. Mapper und Responses-Übersetzung nahmen nur die Textform; ein Objekt ging verloren, die Anfrage lief als `auto`, das Modell durfte antworten, ohne das verlangte Werkzeug aufzurufen. upstream liest `tool_choice` nur als Zeichenkette (`json_value(body, "tool_choice", "auto")`, `server-common.cpp:1282`, b11639; `common_chat_tool_choice_parse_oaicompat` kennt `auto`/`none`/`required`) und fällt bei einem Objekt ebenso auf `auto` zurück. Die Anthropic-Übersetzung bildet ihr `{"type":"tool"}` schon auf `required` ab.
- **Fix:** Ein Objekt mit `type` `function` wird auf beiden Wegen zu `required`, dem nächsten Wert, den llama.cpp kennt (mit einem einzigen Werkzeug genau das verlangte).
- **Reproduktion:** `OpenAiRequestMapperTest#toolChoiceNamingAFunctionRequiresAToolCall`, `ResponsesApiSupportTest#toolChoiceNamingAFunctionRequiresAToolCall`
- **Rot-Grün-Nachweis:** ohne Fix rot, je Weg für sich (2 Fehlschläge, je „Expected: is "required" but: was ""“), mit Fix grün (28/28 und 51/51, skipped 0); Gegenprobe per `git stash` beider Produktionsdateien rot (1 + 1), nach `pop` grün. `OpenAiCompatServerHttpTest` 47/47, `AnthropicApiSupportTest` 22/22, `ResponsesStreamTranslatorTest` 5/5, `OllamaApiSupportTest` 32/32, `LlamaArchitectureTest` 13/13, `spotless:apply` und `spotbugs:check` grün.
- **Nachtrag (2026-10-18, nur beschrieben):** Zwei Kommentare neben dem Fix sind nicht nachgezogen: die Klassen-Javadoc von `OpenAiRequestMapper` nennt `tool_choice` unter den Feldern, die „verbatim as raw JSON“ weitergehen (galt schon vorher nur für die Textform), und der Kommentar über dem Werkzeugblock in `ResponsesApiSupport.toOpenAiChatRequest` beschreibt nur die Textform. Reine Doku-Abweichung, nach der Nutzerregel ohne Test nicht nachträglich geändert; Vorschlag: `tool_choice` aus der Verbatim-Aufzählung nehmen und „ein benanntes Werkzeug wird zu `required`“ ergänzen.
- **Status:** gefixt
- **Commit:** java-llama.cpp `b7151fc`

### F111 – java-llama.cpp: Rerank im TEI-Format (`texts`) wird mit 400 abgewiesen
- **Zeit:** 2026-10-18 09:30 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OaiRerankSupport.java:113` (`readDocuments`, `toResponse`/`toTeiResponse` neu), `llama/src/main/java/net/ladenthin/llama/server/LlamaModelBackend.java` (`rerank`)
- **Schwere:** niedrig
- **Beschreibung:** upstreams `/rerank` (`server-context.cpp:5705`, b11639) unterstützt neben dem Jina/Cohere-Format auch Hugging Faces text-embeddings-inference: eine Anfrage mit `texts` statt `documents` wird angenommen und als bloßes Array `[{index, score}]` beantwortet, bestes zuerst, mit `text` bei `return_text` (`format_response_rerank`, `server-common.cpp:1535`). Der Server hier registriert `/rerank` ebenfalls, verlangte aber `documents` und antwortete TEI-Clients (etwa LangChains `HuggingFaceTEI`-Reranker oder Werkzeuge, die TEI als Reranker ansprechen) mit 400.
- **Fix:** `readDocuments` liest `texts`, wenn `documents` fehlt; `toResponse` wählt das Antwortformat nach der Anfrage, `toTeiResponse` liefert das TEI-Array (Text aus dem `document` der nativen Antwort). Die Naht `toResponse` wurde zuerst verhaltensneutral eingezogen (delegiert an `toOaiResponse`), das Rot ist gegen diesen Stand gezeigt.
- **Reproduktion:** `OaiRerankSupportTest#readDocumentsAcceptsTheTeiTexts`, `#teiRequestGetsTheTeiArrayWithScoreLikeUpstream`, `#teiReturnTextAddsTheDocumentText`
- **Rot-Grün-Nachweis:** gegen den Stand mit der neutralen Naht rot, jeder Teil für sich (3 Fehlschläge: „Unexpected exception thrown: … 'documents' must be a non-empty array“, „Expected: is <true> but: was <false>“, „Expected: is "b" but: was ""“), mit Fix grün (13/13 der Klasse, skipped 0). Die `git stash`-Gegenprobe ging hier nicht (ohne Naht kompiliert der Test nicht; der danach gelesene Bericht war der alte) und wurde durch den Lauf gegen den Naht-Stand ersetzt. `OpenAiCompatServerHttpTest` 47/47, `AnthropicApiSupportTest` 22/22, `ResponsesApiSupportTest` 51/51, `OllamaApiSupportTest` 32/32, `OpenAiRequestMapperTest` 28/28, `LlamaArchitectureTest` 13/13, `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `2831610`

### F112 – java-llama.cpp: Unterpfade einer Route laufen durch deren Handler (`/v1/messages/count_tokens` erzeugt eine ganze Antwort)
- **Zeit:** 2026-10-18 09:50 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiCompatServer.java:271` (`register`, `exactPath` neu)
- **Schwere:** mittel
- **Beschreibung:** Gefunden beim Abgleich mit upstreams Routenliste (`tools/server/server.cpp`, b11639). `com.sun.net.httpserver` ordnet eine Anfrage dem Kontext mit dem **längsten Präfix** zu, nicht dem exakten Pfad, und kein Handler prüfte den Pfad nach. Damit beantwortete jede Route auch alle Pfade, die mit ihr beginnen: `POST /v1/messages/count_tokens` (Claude Code zählt dort Token vor dem Senden) lief als vollständige Nachricht – eine Generierung bis zu 4096 Token (F96) statt eines Fehlers; ebenso upstreams `/v1/chat/completions/input_tokens` und `/v1/responses/input_tokens`, und `/v1/modelsXYZ` lieferte die Modellliste. upstream ordnet exakt zu.
- **Fix:** `register` umhüllt jeden Handler außer dem Sammel-Kontext `/` mit `exactPath`: nur der Pfad selbst (und mit einem abschließenden Schrägstrich) erreicht den Handler, jeder andere bekommt 404. `/v1/reranking`, das nur über das Präfix von `/v1/rerank` erreichbar war, ist jetzt eigens registriert wie bei upstream.
- **Reproduktion:** `OpenAiCompatServerHttpTest#aSubPathOfARouteIsNotHandledByTheRoute`; `#upstreamsRerankingAliasStillAnswers` schützt den Alias und den abschließenden Schrägstrich (ohne Fix grün)
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: is <404> but: was <200>“), mit Fix grün (49/49 der Klasse, skipped 0); Gegenprobe per `git stash` rot (1, Bericht vor dem Lauf gelöscht), nach `pop` grün. Alle 25 Testklassen des Pakets `server` grün, `LlamaArchitectureTest` 13/13, `spotless:apply` und `spotbugs:check` grün. Der Agent (`llama-atmosphere-agent`) ruft keine Unterpfade auf.
- **Offen (nicht geändert):** upstream bietet `/v1/messages/count_tokens` und die `input_tokens`-Routen tatsächlich an; sie hier nachzubauen bräuchte eine Tokenzählung über das Backend (`OpenAiBackend` hat keine) – ein API-Zusatz.
- **Status:** gefixt
- **Commit:** java-llama.cpp `609b53e`

### F113 – java-llama.cpp: `/v1/models` lässt das Pflichtfeld `created` weg
- **Zeit:** 2026-10-18 10:05 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiSseFormatter.java:142` (`modelsJson`)
- **Schwere:** niedrig
- **Beschreibung:** Das OpenAI-Modellobjekt hat vier Pflichtfelder (`id`, `object`, `created`, `owned_by`); `modelsJson` setzte `created` nicht. Streng typisierte Clients lesen die Liste dann nicht ein (etwa Rusts `async-openai`, dessen `Model` `created: u32` verlangt, oder Clients, die ihre Modelle mit dem OpenAI-Schema validieren). upstreams `get_res_model_info` (`server-context.cpp:5078`, b11639) setzt `created` auf die aktuelle Unix-Zeit.
- **Fix:** `created` wird mit der aktuellen Unix-Zeit in Sekunden gesetzt, wie upstream.
- **Reproduktion:** `OpenAiSseFormatterTest#modelsJsonCarriesTheRequiredCreatedTimestampLikeUpstream`
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: is <true> but: was <false>“ für `isIntegralNumber`), mit Fix grün (15/15 der Klasse, skipped 0); Gegenprobe per `git stash` rot (1), nach `pop` grün. `OpenAiCompatServerHttpTest` 49/49, `OaiRerankSupportTest` 13/13, `AnthropicApiSupportTest` 22/22, `ResponsesApiSupportTest` 51/51, `OllamaApiSupportTest` 32/32, `OpenAiRequestMapperTest` 28/28, `LlamaArchitectureTest` 13/13, `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `9c165a8`

### F114 – java-llama.cpp: `/api/version` meldet 0.1.0, VS Code Copilots Ollama-Anbieter lehnt den Server ab
- **Zeit:** 2026-10-18 10:20 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OllamaApiSupport.java:36` (`OLLAMA_VERSION`)
- **Schwere:** mittel
- **Beschreibung:** Die Ollama-Routen sind laut CLAUDE.md und Klassen-Javadoc für Copilots Ollama-Anbieter gedacht (`/api/show` meldet dafür Fähigkeiten und Kontextlänge). Dieser Anbieter (`microsoft/vscode-copilot-chat`, `src/extension/byok/vscode-node/ollamaProvider.ts`) prüft vor jeder Modelloperation `GET /api/version` gegen `MINIMUM_OLLAMA_VERSION = '0.6.4'` (Teile numerisch verglichen, nur `0.0.0` gilt als Entwicklerversion) und bricht sonst mit „Ollama server version … is not supported“ ab. Der Server meldete `0.1.0` – der ganze Copilot-Weg scheiterte an dieser Prüfung. Die Javadoc der Konstante behauptete, Clients prüften nur, dass eine Version kommt.
- **Fix:** `OLLAMA_VERSION` ist `0.9.0`, die erste Ollama-Version mit `think` (die neueste Anfragefunktion, die die Routen seit F78/F97 übersetzen); die Javadoc nennt jetzt den Grund.
- **Reproduktion:** `OllamaApiSupportTest#versionPassesCopilotsMinimumOllamaVersion` (Copilots Vergleich nachgebildet)
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: is <true> but: was <false>“), mit Fix grün (33/33 der Klasse, skipped 0); Gegenprobe per `git stash` rot (1), nach `pop` grün. `OpenAiCompatServerHttpTest` 49/49, `OpenAiSseFormatterTest` 15/15, `OaiRerankSupportTest` 13/13, `AnthropicApiSupportTest` 22/22, `ResponsesApiSupportTest` 51/51, `OpenAiRequestMapperTest` 28/28, `LlamaArchitectureTest` 13/13, `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `fbffef0`

### F115 – java-llama.cpp: `/api/show` verschweigt Continue die Kontextlänge
- **Zeit:** 2026-10-18 10:35 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OllamaApiSupport.java:89` (`showJson`)
- **Schwere:** niedrig
- **Beschreibung:** Fortsetzung von F114 (Quelltext der genannten Clients lesen). Continues Ollama-Anbieter (`continuedev/continue`, `core/llm/llms/Ollama.ts`) liest die Kontextlänge eines Modells ausschließlich aus dem Text `parameters` von `/api/show` (Zeilen `<Schlüssel> <Wert>`, Ollamas Modelfile-Form, `num_ctx`), nicht aus `model_info`. `showJson` lieferte `parameters: ""`; Continue rechnete daher mit seiner eigenen Vorgabe statt mit dem Fenster des Servers – bei einem kleineren `-c` schickt es zu lange Prompts, bei einem größeren nutzt es das Fenster nicht.
- **Fix:** `parameters` trägt `num_ctx <Kontextlänge>` (dieselbe Zahl wie `model_info["llama.context_length"]`).
- **Reproduktion:** `OllamaApiSupportTest#showParametersCarryTheContextLengthWhereContinueReadsIt` (mit Continues regulärem Ausdruck)
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: is "4096" but: was null“), mit Fix grün (34/34 der Klasse, skipped 0); Gegenprobe per `git stash` rot (1), nach `pop` grün. Breite Liste wie F114 grün, `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** (folgt)

## Offene Verdachte ohne Fund-ID

- *(unsicher/niedrig, nur beschrieben, 06:38 UTC)* java-llama.cpp `llama-ffm/.../ffm/JllamaNative.java` `takeStrings`: Bei `count == 0` kehrt die Methode vor dem `try` zurück und gibt das Array nicht frei, das `jllama_rpc_server_devices` auch für 0 Einträge per `calloc(max(n,1))` anlegt (8 Byte Leck je Aufruf). `rpc::server_devices()` fällt aber immer auf das CPU-Gerät zurück, leer wird die Liste nur ohne registriertes CPU-Backend (kaputte Installation) – praktisch unerreichbar. Fix wäre: bei Nicht-NULL-Zeiger immer `STRINGS_FREE` aufrufen. Nicht ohne native Bibliothek + JDK-22-Toolchain testbar, daher kein Commit. Alle übrigen Freigabe-Funktionen (`takeBytesAsString` → `jllama_string_free` laut Header) stimmen mit `jllama_api.h` überein. **Blocker (Stand 2026-10-17):** Der Rot-Grün-Zyklus braucht ein JDK 22+ (Toolchain des Moduls) und ist in dieser Sandbox nicht herstellbar: GitHub-Releases (Adoptium) und api.adoptium.net sind über den Proxy gesperrt. Vorschlag für den Test: `takeStrings` mit der Freigabefunktion als Parameter (paketintern), Aufruf mit Nicht-NULL-Zeiger und Anzahl 0, Erwartung genau ein Freigabeaufruf; läuft dann im CI-Job `test-java-llama-ffm`.
- *(niedrig, unsicher, nur beschrieben, 2026-10-17)* java-llama.cpp `server/OllamaApiSupport`: `done_reason` ist zwischen den Pfaden uneinheitlich. Nicht gestreamtes `/api/chat` reicht OpenAIs `finish_reason` durch (`stop`, `length`, aber auch `tool_calls`, das Ollama selbst nicht kennt – es meldet bei Werkzeugaufrufen `stop`). Gestreamtes `/api/chat` (`toOllamaDoneLine`) und beide `/api/generate`-Formen melden immer `stop`, auch wenn `num_predict` die Antwort abgeschnitten hat. Ein Client, der über `done_reason == "length"` Abschneiden erkennt, sieht es im Streaming nie. Ob das relevant ist, hängt vom Client ab; Vorschlag: `tool_calls` → `stop` abbilden und den `finish_reason` des letzten Stream-Chunks an `toOllamaDoneLine` durchreichen. Kein Fix ohne Entscheidung. Nachtrag 2026-10-18 (Tokenzahlen erledigt durch F98, `done_reason` bleibt offen): Die gestreamte Abschlusszeile (`toOllamaDoneLine`) trägt außerdem weder `prompt_eval_count` noch `eval_count`, die nicht gestreamte Antwort schon (`OllamaApiSupport.java:261`). Anders als die Anthropic- und Responses-Routen (`withUsageChunk`, `OpenAiCompatServer.java:653`/`:741`) fordert `streamOllamaChat` den Usage-Chunk auch nicht an; ein Fix wäre `withUsageChunk` plus Erfassen der Zählung im Akkumulator, Clients, die Token/s aus `eval_count` berechnen, sehen sonst nichts.
- *(Designabweichung, Entscheidung des Nutzers, 2026-10-17)* `OpenAiCompatServer`: Jede Streaming-Route sendet `200` + SSE-Header, **bevor** das Backend läuft. Eine abgelehnte Streaming-Anfrage (leere `messages`, Prompt über der Kontextgröße, `InvalidRequestException`) kommt deshalb als `200` mit einem `error`-Datenereignis an, während dieselbe Anfrage ohne `stream` `400` liefert. Upstream (`server-context.cpp:4906`, b11639) holt vor dem Streamen das erste Ergebnis und antwortet bei einem Fehler mit dessen Status (`res->error`). Bewusst festgeschrieben in `OpenAiCompatServerHttpTest#nativeRejectionOfAStreamedChatCompletionIsAnInvalidRequestEvent` („the headers are out before the backend runs“); Clients, die nur den Status prüfen, sehen eine leere Antwort. Eine Angleichung hieße, die Header erst mit dem ersten Chunk zu senden (Umbau aller fünf Streaming-Routen), daher kein stiller Fix.
- *(niedrig, nur beschrieben, 2026-10-17)* `RouterClient.findModel`/`awaitModelLoaded` vergleichen nur `id`. Upstream löst Modellnamen auch über Aliase auf (`--alias` im Preset, `server_models::get_meta`, `server-models.cpp:1084`) und listet sie unter `aliases`; `awaitModelLoaded("<alias>")` meldet deshalb „Router does not list model“, obwohl der Router das Modell unter diesem Namen lädt und bedient. `RouterModel` liest `aliases` nicht ein; ein Fix bräuchte das Feld (API-Erweiterung), daher hier nur notiert.
- *(niedrig, nur beschrieben, CI-Bedarf: Modell, 2026-10-18, Musterdurchsicht nach F70)* `OpenAiCompatServer.main` → `OpenAiServerCli.toServerConfig(...)` ohne `-c`: beworben werden die festen Vorgaben `DEFAULT_MAX_INPUT_TOKENS` 8192 + `DEFAULT_MAX_OUTPUT_TOKENS` 2048 (Kontextlänge 10240 in `/api/show`/`/props`), unabhängig vom geladenen Modell. Upstreams Vorgabe für `-c` ist 0 = Trainingskontext des Modells (ggf. von `common_fit_params` verkleinert); ein Modell mit kleinerem Fenster (z. B. 2048) wird so zu groß beworben. Der Starter hält das geladene Modell schon in der Hand (Zeile 1102) und könnte das echte Slot-Fenster lesen (`getMetricsTyped()` → `SlotMetrics.n_ctx`); ohne Modell hier nicht testbar.
- *(außerhalb des Ziels – C++, niedrig, nur beschrieben, 2026-10-18, Musterdurchsicht nach F71)* Die JNI-Schalen geben Fehlermeldungen per `env->ThrowNew(cls, jllama_last_error())` (`jllama.cpp:199`/`866`, `native_server.cpp:74`, `rpc_bridge.cpp:42`, `jni_helpers.hpp:120/167`) und `train_engine.cpp:150` per `NewStringUTF(jllama_last_error())` zurück. Beide erwarten Modified UTF-8. Die Meldung kann aber Nutzertext in Standard-UTF-8 enthalten (ein Modellpfad mit Emoji → 4-Byte-Folge) oder sogar rohe ungültige Bytes (nlohmanns `parse_error` zitiert `last read: '…'` byteweise). HotSpot dekodiert das nachsichtig zu Ersatzzeichen, `-Xcheck:jni` warnt, Android-CheckJNI bricht ab. Abhilfe wäre der vorhandene Weg über `utf8_to_jstring_impl` plus `Throw` eines selbst gebauten Exception-Objekts; Java kann es nicht reparieren, die Meldung kommt schon verfälscht an.
- *(niedrig, unsicher, CI-Bedarf: Modell mit Werkzeugaufruf ohne Parameter, 2026-10-18)* Leere Werkzeug-Argumente im Gesprächsverlauf: Für Vorlagen mit `supports_object_arguments` (Qwen, Llama 3 u. a.) parst upstream jedes `function.arguments` mit `json::parse` (`workaround::func_args_not_string`, `common/chat.cpp:1026`, b11639), ein leerer String lässt die ganze Anfrage mit „Failed to parse tool call arguments as JSON“ scheitern. Auf Java-Seite kann `""` entstehen: `ChatResponseParser.parseToolCalls` übernimmt ein textuelles `""` unverändert (und ein fehlendes `arguments` würde `MissingNode.toString()`), `ToolCallDeltaAccumulator`/`StreamingChunkAssembler` setzen `arguments` nur aus den Deltas zusammen, und `LangChain4jMapping.toJllamaAssistantMessage` ersetzt nur `null` durch `"{}"`, nicht `""`. Ob upstream für einen parameterlosen Aufruf `""` statt `"{}"` liefert, hängt vom Parser des Modells ab und ist ohne Modell nicht prüfbar. Billige Absicherung: `ParameterJsonSerializer.buildMessages` schreibt bei leerem `getArgumentsJson()` `"{}"`. Nachtrag (Durchsicht der Übersetzungsschichten): auch auf dem Hinweg kann `""` entstehen – `ResponsesApiSupport.java:126` übernimmt das `arguments` eines `function_call`-Eingabeelements per `asText("")`, `AnthropicApiSupport.java:172` macht aus einem `tool_use`-Block ohne `input` `MissingNode.toString()` = `""` (gemessen); der Rückweg zu Ollama und Anthropic fängt `""` dagegen ab (F37/F38). Eine Normalisierung `""` → `"{}"` an einer Stelle vor upstream (OpenAI-Kern) würde alle Wege decken.
- *(niedrig, nur beschrieben, 2026-10-18)* Rerank gegen upstream: `OaiRerankSupport.toOaiResponse` liefert bei `top_n: 0` alle Ergebnisse, upstream (`format_response_rerank`, `elements.resize(min(top_n, size))`) keines; ein negatives `top_n` ergibt bei uns ebenfalls alle. Und `usage` (`prompt_tokens`/`total_tokens` aus `tokens_evaluated`) fehlt in unserer Antwort ganz, upstream setzt es immer. Kein Absturz und keine falsche Rangfolge, daher nur notiert; `top_n: 0` als „alle“ ist eine vertretbare Lesart, die Abweichung sollte aber bewusst entschieden werden.
- *(erledigt durch F96, 2026-10-18)* Anthropic-Rest nach F76: upstreams `convert_anthropic_to_oai` setzt `max_tokens` 4096, wenn es fehlt (bei uns unbegrenzt; das Protokoll verlangt das Feld, Clients senden es), und reicht `chat_template_kwargs` durch, das wir verwerfen.
- *(niedrig, nur beschrieben, 2026-10-18)* `/props` (`OpenAiSseFormatter.propsJson`) meldet `total_slots` fest als 1; upstream (`get_res_props`, `server-context.cpp:5141`) meldet `params.n_parallel`. Mit `--parallel 4` sieht ein Client einen Slot, obwohl vier Anfragen gleichzeitig laufen können. Die Slotzahl kennt nur `OpenAiServerCli`, nicht `OpenAiServerConfig`; ein Fix bräuchte ein neues Konfigurationsfeld. Kein bekannter Client richtet sich danach, deshalb nur notiert.
- *(erledigt durch F97, 2026-10-18)* Ollama-Antworten tragen den Denkinhalt nicht: Ollama legt ihn in `message.thinking` ab (blockierend und je Stream-Zeile), `OllamaApiSupport` übernimmt aus OpenAIs Antwort nur `content` und `tool_calls`, `reasoning_content` geht verloren. Ein Client mit `think: true` sieht keine Gedanken. Eine Ergänzung der Antwortform, kein Fehlverhalten der Anfrage; daher nur notiert.
- *(mittel, teilweise gefixt durch F95, 2026-10-18)* **Nachtrag:** Die Sampling-Felder mit vorhandenem Setter übernimmt seit F95 der Mapper (beide Chat-Pfade und das gestreamte Completion); offen bleiben `logit_bias`, `n_probs`/`logprobs` und `n`. Ursprünglicher Text: Gestreamtes `/v1/completions` verliert weiter alle Felder außerhalb von `applyCommonFields`, die der nicht gestreamte Weg durchreicht und upstream liest: `n_predict`, `min_p`, `repeat_penalty`, `grammar`, `json_schema`, `logit_bias`, `n_probs`/`logprobs` und weitere. Dieselbe Anfrage erzeugt gestreamt also andere Ausgaben. Ein Fix bräuchte einen Streaming-Einstieg mit rohem JSON (die Natives `requestCompletion`/`receiveCompletionJson` sind paketintern in `net.ladenthin.llama`, `LlamaModelBackend` liegt in `server`) – ein Umbau der API, daher zur Entscheidung des Nutzers. Dasselbe gilt in kleinerem Maß für den Chat-Pfad (blockierend und gestreamt), der seit F77 die Reasoning-Steuerungen übernimmt, aber z. B. `min_p`, `repeat_penalty`, `logprobs` und `n_predict` weiter verwirft. Ebenso `n` (mehrere Antworten): upstream liest es als Alias von `n_cmpl` (`server-schema.cpp:62`, Grenze `n_parallel`), Java kennt das Feld gar nicht (kein `RequestField`, kein Setter), eine Anfrage mit `n: 3` bekommt still eine Antwort. Ein Fix bräuchte einen neuen `RequestField`-Eintrag (C++-Vertragstest `test_wire_contracts.cpp`, CI-Bedarf) und einen Setter – ein API-Zusatz.
- *(niedrig, nur beschrieben, 2026-10-18)* Responses-`refusal`-Teile in Assistenten-Nachrichten des Verlaufs verwirft `ResponsesApiSupport` (der Teil hat `refusal`, kein `text`). Upstream übersetzt sie zu einem Teil vom Typ `refusal`, den sein eigener Chat-Parser dann mit 400 abweist (`common/chat.cpp:373`, nur `text`/`media_marker`). Beides ist unbefriedigend; als Text zu übernehmen wäre die nützlichste Lesart. Entscheidung des Nutzers.
- *(niedrig, nur beschrieben, 2026-10-18)* Anthropic `stop_sequence` ist bei uns immer `null`; upstreams `to_json_anthropic` (`server-task.cpp:778`) setzt das getroffene Stoppwort (bei `stop_reason` `end_turn`). Die OpenAI-Antwort, aus der `/v1/messages` übersetzt, trägt das Stoppwort nicht; ein Fix bräuchte es aus der nativen Antwort (`stopping_word` der Nicht-OAI-Form). Sonst stimmen blockierender und gestreamter Pfad überein (Usage mit `cache_read_input_tokens`, `stop_reason`-Abbildung).
- *(erledigt durch F106, 2026-10-18)* Responses-Felder: upstreams `convert_responses_to_chatcmpl` kopiert den **ganzen** Körper in die Chat-Anfrage (`chatcmpl_body = response_body`, nur `input`/`instructions`/`max_output_tokens` umbenannt bzw. entfernt), sodass `top_k`, `seed`, `chat_template_kwargs`, `min_p`, `repeat_penalty` & Co. ankommen. `ResponsesApiSupport.toOpenAiChatRequest` übernimmt nur `temperature`, `top_p`, `max_output_tokens` und `reasoning.effort`; der Rest geht still verloren (Muster F74/F104). Nächste Runde: die Felder, die der Mapper liest (F95), aus der Responses-Anfrage übernehmen und je Feld einen Test schreiben.
- *(niedrig, nur beschrieben, Entscheidung des Nutzers, 2026-10-18)* Continue entscheidet die FIM-Fähigkeit eines Ollama-Modells daran, ob das `template` aus `/api/show` die Variable `.Suffix` enthält (`Ollama.ts`, „There is no API to get the model's FIM capabilities“). `showJson` liefert `template: ""`, meldet aber `insert` unter `capabilities` und bedient `suffix` in `/api/generate`; Continue nutzt daher für Autovervollständigung seine eigenen Prompt-Vorlagen statt des FIM-Wegs. Ein Platzhalter-Template mit `{{ .Suffix }}` würde das ändern, wäre aber eine erfundene Vorlage, und der Server weiß nicht, ob das Modell FIM-Token hat. Kein Fix ohne Entscheidung.
