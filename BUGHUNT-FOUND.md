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
- [F16](#f16) — mittel — *gefixt (JNI; die FFM-Hälfte als F198)*
- [F17](#f17) — mittel — *gefixt*
- [F18](#f18) — mittel — *gefixt (Fix `c1e5dcd`; roter Regressionstest nachgereicht, bis dahin „nur beschrieben“, siehe `209ee6c`)*
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
- [F57](#f57) — niedrig — *gefixt*
- [F59](#f59) — niedrig — *gefixt*
- [F60](#f60) — niedrig — *gefixt*
- [F61](#f61) — niedrig — *gefixt*
- [F62](#f62) — niedrig — *gefixt als F94 (Duplikat: derselbe Befund wurde in der Musterdurchsicht nach F60/F61 noch einmal aufgenommen und dort behoben; Nachtrag 2026-10-19)*
- [F63](#f63) — niedrig — *gefixt*
- [F64](#f64) — mittel — *gefixt*
- [F65](#f65) — niedrig — *gefixt*
- [F67](#f67) — niedrig — *gefixt (zuerst Dokumentation, seit 2026-10-19 Patch 0021, siehe Nachtrag)*
- [F68](#f68) — niedrig — *gefixt*
- [F69](#f69) — niedrig — *gefixt (Dokumentation)*
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
- [F116](#f116) — niedrig — *gefixt*
- [F117](#f117) — mittel — *gefixt*
- [F118](#f118) — niedrig — *gefixt*
- [F123](#f123) — niedrig — *gefixt*
- [F125](#f125) — niedrig — *gefixt*
- [F128](#f128) — mittel — *gefixt*
- [F133](#f133) — niedrig — *gefixt*
- [F134](#f134) — mittel — *gefixt*
- [F137](#f137) — niedrig — *gefixt*
- [F138](#f138) — niedrig — *gefixt*
- [F143](#f143) — niedrig — *gefixt*
- [F158](#f158) — niedrig bis mittel — *gefixt*
- [F165](#f165) — mittel — *gefixt*
- [F166](#f166) — niedrig — *gefixt (Java-API-Pfad; Server-Übersetzungen als F172)*
- [F167](#f167) — niedrig bis mittel — *gefixt*
- [F168](#f168) — niedrig — *gefixt*
- [F172](#f172) — niedrig — *gefixt*
- [F173](#f173) — niedrig — *gefixt*
- [F174](#f174) — niedrig — *gefixt*
- [F175](#f175) — niedrig — *gefixt*
- [F176](#f176) — niedrig — *gefixt*
- [F177](#f177) — niedrig — *gefixt*
- [F178](#f178) — niedrig — *gefixt*
- [F179](#f179) — niedrig — *gefixt*
- [F180](#f180) — mittel — *gefixt (der Rest unter (14), Wahrscheinlichkeiten im gestreamten `/v1/completions`, mit F186)*
- [F181](#f181) — niedrig bis mittel — *gefixt*
- [F182](#f182) — mittel — *gefixt*
- [F183](#f183) — niedrig — *gefixt*
- [F184](#f184) — mittel — *gefixt*
- [F185](#f185) — niedrig bis mittel — *gefixt*
- [F186](#f186) — mittel — *gefixt*
- [F187](#f187) — mittel — *gefixt*
- [F188](#f188) — niedrig bis mittel — *gefixt*
- [F189](#f189) — niedrig bis mittel — *gefixt*
- [F190](#f190) — niedrig — *gefixt*
- [F197](#f197) — mittel — *gefixt*
- [F198](#f198) — mittel — *gefixt*
- [F199](#f199) — mittel — *gefixt*
- [F200](#f200) — niedrig bis mittel — *gefixt*
- [F201](#f201) — niedrig — *gefixt*
- [F202](#f202) — niedrig — *gefixt*
- [F203](#f203) — niedrig — *gefixt*
- [F204](#f204) — niedrig bis mittel — *gefixt*
- [F205](#f205) — niedrig — *gefixt*

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
- **Status:** gefixt (JNI; die FFM-Hälfte als F198)
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

<a id="f56"></a>
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
- **Historie:** Die Begründung „upstream does not currently expose a cumulative logical or cached prompt total“ steht in der erreichbaren Historie schon vor #494 (2026-10-09, flacher Klon, älter nicht sichtbar); `getCumulativeCachedPromptTokens()` und die Weitergabe von `n_prompt_tokens_cached_total` kamen mit `1eef547` (Bump auf b11538, 2026-10-09) – seitdem ist die Begründung überholt, `getCumulativeUsage()` wurde aber nicht nachgezogen. Bei b11639 nachgeprüft: upstream zählt in `prompt` nur ausgewertete Tokens und die Cache-Treffer getrennt (`server-common.h`, „cached ones are counted separately below“).
- **Fix:** Entscheidung des Nutzers: `new Usage(processed + cached, predicted, cached)`; Javadoc korrigiert, `getWindowUsage()` dokumentiert, warum dort kein Cache-Anteil steht (upstream führt für das Fenster keinen). CHANGELOG unter „Fixed“ mit dem Hinweis für Aufrufer, die `getPromptTokens()` als „ausgewertet“ gelesen haben.
- **Tests:** `ServerMetricsTest#cumulativeUsage` (schrieb das alte Verhalten fest, jetzt 800/700/100/1000 aus 100 ausgewerteten + 700 gecachten) und neu `#cumulativeUsage_readsLikeTheUsageOfAChatResponse` (10 + 4 → `new Usage(14, 3, 4)`).
- **Rot-Grün-Nachweis:** ohne Fix rot (2 von 23), mit Fix grün (23 von 23).
- **Status:** gefixt
- **Commit:** java-llama.cpp `a529c5c`

<a id="f59"></a>
### F59 – java-llama.cpp: `FfmNativeServer.isRunning()` kann nach einem parallelen `close()` freigegebenen Speicher lesen
- **Zeit:** 2026-10-17 21:25 UTC
- **Datei:** `llama-ffm/src/main/java/net/ladenthin/llama/ffm/FfmNativeServer.java` (`isRunning`, `close`)
- **Schwere:** niedrig
- **Beschreibung:** `isRunning()` prüft `closed` und ruft dann `jllama_server_is_running(handle)`. `close()` setzt `closed` und ruft `jllama_server_stop`, das laut `jllama_api.h` den Handle freigibt. Läuft `close()` (etwa aus einem Shutdown-Hook) zwischen der Prüfung und dem nativen Aufruf eines anderen Threads, liest `jllama_server_is_running` einen freigegebenen `jllama_server` – undefiniertes Verhalten im nativen Code, schlimmstenfalls ein JVM-Absturz. Der JNI-Zwilling `server.NativeServer` hält genau deshalb `isRunning()` und `close()` unter einem gemeinsamen `lifecycleLock`; die FFM-Klasse dokumentiert keine Einschränkung. (`FfmLlamaModel` hat dasselbe Muster, schreibt aber ausdrücklich, dass `close()` keinen neuen Aufruf überholen darf – dort ein dokumentierter Vertrag, kein Fund.)
- **Messung (J7, `BUGHUNT-MEASUREMENTS.md` in `b689d7b`, JDK 25):** mit einer Naht rot – `close()` gab den Handle frei, während `isRunning()` noch im nativen Aufruf war; ohne Naht 862 Mio. Aufrufe in 30 Runden ohne sichtbaren Schaden (das Fenster ist schmal).
- **Fix:** Entscheidung des Nutzers: das Lock aus `NativeServer`. `isRunning()` und `close()` laufen unter `lifecycleLock`, `closed` ist ein einfaches Feld unter diesem Lock. Die beiden nativen Aufrufe liegen hinter der paketinternen Naht `FfmNativeServer.Backend` (wie `NativeServer.Backend`), statt veränderlicher statischer Method-Handles.
- **Tests:** `FfmNativeServerTest` (4, ohne Bibliothek): `close()` wartet auf ein `isRunning()` im nativen Aufruf und gibt erst danach frei; `isRunning()` nach `close()` fragt den Handle nicht ab; zweites `close()` ohne Wirkung; `isRunning()` fragt im offenen Zustand.
- **Rot-Grün-Nachweis:** lokal mit JDK 25 (Temurin 25.0.1, Toolchain): ohne Lock (Naht beibehalten, `AtomicBoolean` wie vorher) rot (1 von 4, `close()` wartete nicht), mit Lock grün.
- **Nachtrag (Branch-Review 2026-10-20):** Dieselbe Prüfen-dann-Rufen-Lücke hatte `FfmTextToSpeech`; dort mit demselben Lock als F198 behoben.
- **Status:** gefixt
- **Commit:** c7c883d

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
- **Status:** gefixt als F94 (Duplikat: derselbe Befund wurde in der Musterdurchsicht nach F60/F61 noch einmal aufgenommen und dort behoben; Nachtrag 2026-10-19)
- **Commit:** java-llama.cpp `ee99d85` (F94)

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
- **Fix:** Entscheidung des Nutzers: Dokumentation korrigieren (ohne Test, Nutzerregel für Doku-Korrekturen). Javadoc von `LlamaModel.setLoraAdapters` und README sagen jetzt, dass die KV-Caches auf diesem Weg **nicht** geleert werden, und nennen `eraseSlot(id)` bzw. `withCachePrompt(false)`. Vor der Änderung am Quelltext von b11639 nachgeprüft: `SERVER_TASK_TYPE_SET_LORA` (`server-context.cpp:3077`) setzt nur `params_base.lora_adapters`, `launch_slot_with_task` (`:1868`) übernimmt sie im `else`-Zweig ohne `prompt.clear()`. Das Leeren in `engine::set_lora_adapters` (C++) bleibt als Möglichkeit, ist aber nicht beschlossen.
- **Reproduktion:** keine lauffähige: es braucht ein Modell und einen dazu passenden LoRA-Adapter (nicht in `models.csv`), und in dieser Sandbox gibt es keine native Bibliothek und kein `gguf-py`. Ein Test würde denselben Prompt vor und nach `setLoraAdapter(0, 0f)` mit Greedy-Sampling schicken und mit einem frisch geladenen Modell ohne Adapter vergleichen.
- **CI-Bedarf:** ein kleines Modell mit LoRA-Adapter im Modellsatz (oder ein zur Laufzeit erzeugtes Zufallsmodell samt Adapter, wie `test_c_api.cpp` es für das Modell macht) – nur, falls das Verhalten geändert werden soll.
- **Status:** gefixt (zuerst Dokumentation, seit 2026-10-19 Patch 0021, siehe Nachtrag)
- **Commit:** a977a1e
- **Nachtrag 2026-10-19 (Verhalten geändert):** J9 hat F67 bestätigt (`b689d7b`: nach `setLoraAdapter(0, 0f)` lieferte der warme Slot „. He saw a picture of a bird …“ statt „ was a little girl …“ wie ein geleerter Slot). Entscheidung des Nutzers: upstream-fähiger Patch `0021` statt eines Leerens in `engine::set_lora_adapters`, weil upstreams eigene HTTP-Route `POST /lora-adapters` denselben Fehler hat. Der `else`-Zweig von `launch_slot_with_task` wendet jetzt dieselbe Regel an wie LoRA pro Anfrage (`lora_should_clear_cache`, aLoRA behält den Cache). Das deckt auch den zweiten Weg ab, den die Durchsicht fand: eine Anfrage mit eigenen Adaptern füllt den Cache, die nächste ohne läuft mit den Server-Adaptern darauf weiter. Nicht abgedeckt, in Javadoc/README genannt: der Host-RAM-Prompt-Cache (`--cache-ram`, standardmäßig an) speichert Slot-Zustände ohne ihre Adapter. Javadoc `setLoraAdapters`, README, CHANGELOG, CLAUDE.md (Patch- und Testtabelle), `docs/patches.md`. Upstream-master bewusst nicht geprüft (Entscheidung des Nutzers: zeigt sich beim nächsten Bump über den Drop-Check).
- **Tests (Patch):** `src/test/cpp/test_lora_cache.cpp` (4): Zufallsmodell + Zufalls-LoRA auf K, V und FFN-Ausgang, ein Slot, `--cache-ram 0`, greedy; nach Skala 1→0, 0→1 und nach einer Anfrage mit eigenen Adaptern muss der warme Slot wie ein geleerter antworten; ein vierter Fall schließt das leere Bestehen aus (Adapter ändert die Antwort, Wiederholung nutzt den Cache). Das Modellschreiben liegt jetzt in `tiny_llama_gguf.hpp`, gemeinsam mit `test_c_api.cpp`.
- **Rot-Grün-Nachweis (Patch):** ohne Patch rot (3 von 4; z. B. „mYmq5*mq“ statt „C+oM~kM~“), mit Patch grün (4 von 4, drei Läufe); ganze C++-Suite 633/633; alle 21 Patches wenden sich der Reihe nach sauber auf b11639 an; `check-patches.py` 0 Probleme.
- **Commit (Patch):** java-llama.cpp `c4a23c5`

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
- **Beschreibung:** Die Javadoc nennt für den Anfragewert `n_predict` „-1 = infinity, -2 = until context filled“. Upstream (`tools/server/server-schema.cpp:44`, b11639) deklariert das Feld mit `set_hard_limits(-1, INT32_MAX)`, und `is_hard_limit` heißt laut `server-schema.h:52` „throw error if the value is invalid“: `withNPredict(-2)` endet also in einer 400 bzw. `InvalidRequestException`, statt bis zum Kontextende zu generieren. `-2` gibt es nur für die Startoption `-n` – **Nachtrag:** dort ist es für den Server ebenfalls falsch dokumentiert, siehe F158.
- **Fix:** Entscheidung des Nutzers: Dokumentation korrigieren (ohne Test). Die Javadoc von `withNPredict` sagt jetzt: `-1` übernimmt die `--predict`-Einstellung des Modells (selbst standardmäßig unbegrenzt, `server-context.cpp:1964`), der Server nimmt `-1` und größer an, `-2` lehnt er als ungültige Anfrage ab. Upstreams Beschreibung für `0` („no tokens will be generated but the prompt is evaluated“) ist bewusst **nicht** übernommen: im Quelltext findet sich dafür kein eigener Zweig, `has_budget()` stoppt erst nach dem ersten Token; ohne Messung bleibt die Aussage draußen.
- **Reproduktion:** keine lauffähige hier (kein Modell, keine native Bibliothek). Ein Test in einer modellgestützten Klasse: `model.complete(InferenceParameters.empty().withPrompt("x").withNPredict(-2))` → `InvalidRequestException` mit `n_predict` in der Meldung.
- **CI-Bedarf:** ein modellgestützter Java-Test (jedes Modell aus `models.csv` genügt), falls `-2` zusätzlich in Java abgewiesen werden soll.
- **Nachtrag 2026-10-19 (geschlossen):** J2a hat die Ablehnung gemessen (`InvalidRequestException`, „Field 'n_predict': Value must be between -1 <= value <= 2147483647“). Entscheidung des Nutzers: kein Java-Check (er würde denselben Fehler nur früher melden und upstreams Grenze ein zweites Mal pflegen), dafür die Ablehnung modellgestützt festgeschrieben: `ErrorHandlingTest#testNPredictMinusTwoPerRequestIsAnInvalidRequest` (läuft in jedem `test-java-*`-Job mit dem CI-Modell). Lokal mit einem zufälligen Zwei-Schicht-GGUF geprüft: mit `-2` grün, als Gegenprobe mit `-1` rot („Expected InvalidRequestException to be thrown, but nothing was thrown“).
- **Status:** gefixt (Dokumentation)
- **Commit:** a977a1e

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

<a id="f92"></a>
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

<a id="f93"></a>
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

<a id="f94"></a>
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

<a id="f95"></a>
### F95 – java-llama.cpp: gestreamtes `/v1/completions` und der Chat-Pfad verwerfen llama.cpps Sampling-Felder
- **Zeit:** 2026-10-18 05:30 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiRequestMapper.java:200` (`applyLlamaCppSampling`, neu; `readMaxTokens`)
- **Schwere:** mittel
- **Beschreibung:** Aus dem offenen Verdacht vom 2026-10-18 (gestreamtes `/v1/completions`). Das nicht gestreamte `/v1/completions` reicht den Körper an upstreams Schema durch (`server-schema.cpp`, b11639), das gestreamte baut `InferenceParameters` im `OpenAiRequestMapper` aus einer festen Feldliste. Alles außerhalb dieser Liste ging verloren: `min_p`, `typical_p`, `repeat_penalty`, `repeat_last_n`, `top_n_sigma`, `xtc_*`, `dynatemp_*`, `min_keep`, `n_keep`, `ignore_eos`, `grammar`/`json_schema`, und `n_predict` (upstreams Name, `max_tokens`/`max_completion_tokens` sind seine Aliase und verlieren gegen ihn). Dieselbe Anfrage erzeugte gestreamt also andere Ausgaben, eine Grammatik wirkte nur ohne `stream`. Derselbe Mapper bedient beide Chat-Pfade, die Felder fehlten dort ebenso. Alle Felder haben bereits einen Setter; anders als im Verdacht befürchtet, braucht dieser Teil keinen API-Umbau.
- **Fix:** `applyLlamaCppSampling` (aus `applyCommonFields`) übernimmt die Felder mit den vorhandenen Settern; `json_schema` nur ohne `grammar`, wie upstream. `readMaxTokens` liest zuerst `n_predict`. Weiter offen (Verdacht bleibt): `logit_bias`, `n_probs`/`logprobs` (im gestreamten Completion-Chunk ohnehin nicht ausgegeben) und `n`.
- **Reproduktion:** `OpenAiRequestMapperTest#llamaCppSamplingExtensionsReachTheStreamedCompletion`, `#nPredictWinsOverMaxTokensLikeUpstream`, `#jsonSchemaReachesTheStreamedCompletionUnlessAGrammarIsGiven`
- **Rot-Grün-Nachweis:** ohne Fix rot (3 Fehlschläge: „Expected: is a numeric value within <1.0E-4> of <0.05> but: <0.0>“, „Expected: is <7> but: was <50>“, „Expected: is "object" but: was ""“), mit Fix grün (25/25 der Klasse, skipped 0); Gegenprobe per `git stash` rot (3), nach `pop` grün. `OpenAiCompatServerHttpTest` 46/46, `AnthropicApiSupportTest` 19/19, `ResponsesApiSupportTest` 18/18, `OllamaApiSupportTest` 24/24, `LlamaArchitectureTest` 13/13, `spotless:apply` und `spotbugs:check` grün (eine erste Fassung mit allen Feldern in einer Methode meldete `CE_CLASS_ENVY`, eine Map-Tabelle `DMC_DUBIOUS_MAP_COLLECTION`; jetzt je Feld ein Aufruf über `withFloat`/`withInt`).
- **Nachtrag (Branch-Review 2026-10-20):** Das hier „weiter offen“ Genannte ist erledigt: `logit_bias` und `n_probs` mit F180, das gestreamte `/v1/completions` mit F186 (upstreams eigener Stream, Wahrscheinlichkeiten eingeschlossen), `n > 1` mit F180 und auf allen nativen Wegen mit F197.
- **Status:** gefixt
- **Commit:** java-llama.cpp `8ce9ffc`

<a id="f96"></a>
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

<a id="f97"></a>
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

<a id="f98"></a>
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

<a id="f99"></a>
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

<a id="f102"></a>
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

<a id="f103"></a>
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

<a id="f104"></a>
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

<a id="f105"></a>
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

<a id="f106"></a>
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

<a id="f107"></a>
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

<a id="f108"></a>
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

<a id="f109"></a>
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

<a id="f111"></a>
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

<a id="f112"></a>
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

<a id="f113"></a>
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

<a id="f114"></a>
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

<a id="f115"></a>
### F115 – java-llama.cpp: `/api/show` verschweigt Continue die Kontextlänge
- **Zeit:** 2026-10-18 10:35 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OllamaApiSupport.java:89` (`showJson`)
- **Schwere:** niedrig
- **Beschreibung:** Fortsetzung von F114 (Quelltext der genannten Clients lesen). Continues Ollama-Anbieter (`continuedev/continue`, `core/llm/llms/Ollama.ts`) liest die Kontextlänge eines Modells ausschließlich aus dem Text `parameters` von `/api/show` (Zeilen `<Schlüssel> <Wert>`, Ollamas Modelfile-Form, `num_ctx`), nicht aus `model_info`. `showJson` lieferte `parameters: ""`; Continue rechnete daher mit seiner eigenen Vorgabe statt mit dem Fenster des Servers – bei einem kleineren `-c` schickt es zu lange Prompts, bei einem größeren nutzt es das Fenster nicht.
- **Fix:** `parameters` trägt `num_ctx <Kontextlänge>` (dieselbe Zahl wie `model_info["llama.context_length"]`).
- **Reproduktion:** `OllamaApiSupportTest#showParametersCarryTheContextLengthWhereContinueReadsIt` (mit Continues regulärem Ausdruck)
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: is "4096" but: was null“), mit Fix grün (34/34 der Klasse, skipped 0); Gegenprobe per `git stash` rot (1), nach `pop` grün. Breite Liste wie F114 grün, `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `4d45be1`

<a id="f116"></a>
### F116 – java-llama.cpp: Ollama-Werkzeugergebnisse verlieren den Werkzeugnamen
- **Zeit:** 2026-10-18 10:45 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OllamaApiSupport.java:208` (`toOpenAiMessage`)
- **Schwere:** niedrig
- **Beschreibung:** Ollama-Clients (etwa Continues Ollama-Anbieter) schicken Werkzeugergebnisse als `{"role":"tool","content":…,"tool_name":…}`, ohne `tool_call_id`. `toOpenAiMessage` kopierte die Nachricht unverändert; llama.cpps Chat-Parser liest den Namen eines Werkzeugergebnisses aber aus `name` (`common/chat.cpp:426`, b11639) und kennt `tool_name` nicht. Ohne Kennung und ohne Namen konnte die Vorlage nicht sagen, welches Werkzeug geantwortet hat – bei mehreren Aufrufen in einer Runde ordnet das Modell die Ergebnisse nur noch nach der Reihenfolge zu.
- **Fix:** `tool_name` wird zu `name` (ein vorhandenes `name` hat Vorrang).
- **Reproduktion:** `OllamaApiSupportTest#toolResultNameBecomesTheNameLlamaCppReads`
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: is "get_weather" but: was ""“), mit Fix grün (35/35 der Klasse, skipped 0); Gegenprobe per `git stash` rot (1), nach `pop` grün. Breite Liste wie F114 grün, `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `a5e86bf`

<a id="f117"></a>
### F117 – java-llama.cpp: Gestreamte Responses-Fehler erreichen Responses-Clients nicht
- **Zeit:** 2026-10-18 11:20 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiCompatServer.java:782` (`streamResponses`), `ResponsesStreamTranslator.java`
- **Schwere:** mittel
- **Beschreibung:** Scheiterte ein gestreamtes `/v1/responses` (abgelehnte Anfrage, Fehler bei der Inferenz), schrieb der Server nach `response.created` nur `event: error` mit `{"error":{…}}` – ohne `type` im Datenobjekt und ohne `response.failed`. Responses-Clients verteilen nach dem `type` der Daten: Codex (`codex-api/src/sse/responses.rs`, `process_responses_event`) verwirft ein Ereignis ohne `type` beim Parsen und liest die Fehlermeldung nur aus `response.failed` (`responses_error.rs`, `parse_failed_response` → `response.error.message`/`code`); Ergebnis war „stream closed before response.completed“ statt der eigentlichen Meldung, etwa des Kontextüberlaufs.
- **Fix:** `ResponsesStreamTranslator.failed(message, code)` erzeugt ein `response.failed` mit der Antwort (`status: failed`, `error.code`/`error.message`) und fortlaufender `sequence_number`; `streamResponses` schreibt es in beiden Fehlerzweigen vor das bisherige `error`-Ereignis, das für Clients bleibt, die ein `error`-Objekt auf oberster Ebene auswerten (OpenAIs Python-SDK) und upstreams Form entspricht.
- **Reproduktion:** `OpenAiCompatServerHttpTest#rejectedStreamedResponseEndsWithAResponseFailedEvent`, `#failedStreamedResponseEndsWithAResponseFailedEvent`
- **Rot-Grün-Nachweis:** ohne Fix beide rot („Expected: a string containing "event: response.failed" but: was "event: response.created …"“), mit Fix grün (51/51 der Klasse, skipped 0); Gegenprobe per `git stash` beider Produktionsdateien rot (2), nach `pop` grün. Breite Liste (Server-Tests inkl. `ResponsesStreamTranslatorTest`, `LlamaArchitectureTest`) grün, `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `011f5c3`

<a id="f118"></a>
### F118 – java-llama.cpp: Anthropic-Serverfehler heißen `invalid_request_error`
- **Zeit:** 2026-10-18 11:45 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiCompatServer.java:718` (`anthropicError`)
- **Schwere:** niedrig
- **Beschreibung:** `anthropicError` setzte den Fehlertyp immer auf `invalid_request_error`, auch für eine Antwort mit HTTP 500 und für das `error`-Ereignis eines gestreamten `/v1/messages`, das an einem Inferenzfehler scheiterte. Anthropics Fehlertypen trennen die Schuld: `invalid_request_error` ist der Fehler des Clients (400), `api_error` ein unerwarteter Serverfehler (500). Ein Client (Claude Code), der nach dem Typ entscheidet, ob ein Wiederholen lohnt, hielt einen Serverfehler für eine fehlerhafte eigene Anfrage – im Stream kommt der Status 200 ohnehin vor dem Fehler, dort trägt nur der Typ die Unterscheidung.
- **Fix:** `anthropicError(message, type)`; die Ablehnungszweige (`IllegalArgumentException`/`InvalidRequestException`) behalten `invalid_request_error`, die Fehlerzweige melden `api_error`.
- **Reproduktion:** `OpenAiCompatServerHttpTest#aFailedAnthropicMessageIsAnApiError`, `#aFailedStreamedAnthropicMessageIsAnApiError` (dazu als Schutz `#aRejectedStreamedAnthropicMessageStaysAnInvalidRequestError`)
- **Rot-Grün-Nachweis:** ohne Fix beide rot („Expected: a string containing "\"error\":{\"type\":\"api_error\"" but: was "{"type":"error","error":{"type":"invalid_request_error","message":"decoding failed"}}"“), mit Fix grün (54/54 der Klasse, skipped 0); Gegenprobe per `git stash` rot (2), nach `pop` grün. Breite Liste wie F117 grün, `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `ccba21a`

<a id="f123"></a>
### F123 – java-llama.cpp: `logprobs`/`top_logprobs` einer Chat-Anfrage gehen verloren
- **Zeit:** 2026-10-18 14:35 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiRequestMapper.java:72` (`toInferenceParameters`)
- **Schwere:** niedrig
- **Beschreibung:** Eine OpenAI-Chat-Anfrage mit `"logprobs":true` (und optional `"top_logprobs":N`) bekam keine Token-Wahrscheinlichkeiten: Der Mapper übernahm keins der beiden Felder, `InferenceParameters` trug also kein `n_probs`. Upstream bildet sie in `oaicompat_chat_params_parse` (`server-common.cpp:1456`, b11639) auf `n_probs = top_logprobs` ab, Standard 20, und die nativen Chat-Ergebnisse tragen die Wahrscheinlichkeiten dann von selbst (blockierend und im OAI-Chunk-Pfad). Teil des offenen F95-Verdachts („`n_probs`/`logprobs`“), der als Blocker eine API-Erweiterung nannte – `withNProbs` gab es aber schon (Erkenntnis F95: Verdacht in Teile zerlegen). Gefunden per Probe-Klasse über den Mapper.
- **Fix:** Bei `logprobs: true` setzt der Chat-Mapper `withNProbs(top_logprobs)`, ohne `top_logprobs` 20 wie upstream. Nur im Chat-Pfad: Das `logprobs` des alten `/v1/completions` ist eine Zahl mit anderer Bedeutung. Nicht übernommen: upstreams zwei Ablehnungen (`top_logprobs` ohne `logprobs`; `logprobs` mit Werkzeugen im Stream).
- **Reproduktion:** `OpenAiRequestMapperTest#logprobsWithTopLogprobsBecomesNProbs`, `#logprobsWithoutTopLogprobsDefaultsToTwentyLikeUpstream` (dazu als Schutz `#logprobsFalseLeavesNProbsUnset`)
- **Rot-Grün-Nachweis:** ohne Fix beide rot („Expected: is <3> but: was <-1>“ bzw. „Expected: is <20> but: was <-1>“), mit Fix grün (31/31 der Klasse, skipped 0); Gegenprobe per `git stash` rot (2), nach `pop` grün. Breite Liste wie F117 grün, `spotless:apply` und `spotbugs:check` grün. Der erste Grün-Lauf scheiterte am Compiler (`NotJavadoc`: Konstante zwischen fremde Javadoc und Deklaration eingefügt, wie bei F108) und meldete trotzdem den alten roten Bericht – das Prüfskript löscht den Bericht jetzt auch vor dem Grün-Lauf.
- **Status:** gefixt
- **Commit:** java-llama.cpp `3e86360`

<a id="f125"></a>
### F125 – java-llama.cpp: `OpenAiServerCli` nimmt einen Port außerhalb von 0..65535 an; der Fehler kommt erst nach dem Modell-Laden
- **Zeit:** 2026-10-18 15:40 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiServerCli.java:72` (`parse`, `--port`), Aufrufer `OpenAiCompatServer.main`
- **Schwere:** niedrig
- **Beschreibung:** `--port`/`-p` lief nur durch `Integer.parseInt`. `-p 70000` oder `-p -1` wurde angenommen; `OpenAiCompatServer.main` lud dann zuerst das Modell (bei großen GGUFs Minuten) und scheiterte erst beim Binden in `new InetSocketAddress` mit „port out of range“ als nicht gefangener `IllegalArgumentException` samt Stacktrace. Die Javadoc von `main` verspricht für Usage-Fehler eine Meldung auf stderr und Rückkehr; ein nicht numerischer Port wird schon so behandelt, ein numerischer außerhalb des TCP-Bereichs nicht.
- **Fix:** `portValue` prüft 0..65535 (0 bleibt erlaubt: freier Port) und wirft den üblichen Usage-Fehler.
- **Reproduktion:** `OpenAiServerCliTest#portOutsideTheTcpRangeThrows` (`-1`, `65536`, `70000`), Grenzen in `#portAtTheEndsOfTheTcpRangeIsAccepted` (0, 1, 65535)
- **Rot-Grün-Nachweis:** ohne Fix rot in allen drei Fällen („Expected java.lang.IllegalArgumentException to be thrown, but nothing was thrown.“), mit Fix grün (45/45 der Klasse, skipped 0); Gegenprobe per `git stash` rot (3), nach `pop` grün. Breite Liste: `OpenAiCompatServerHttpTest` 54/54, `ServerLauncherTest` 7/7, `LlamaArchitectureTest` 13/13; `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `ba26cdc`

<a id="f128"></a>
### F128 – java-llama.cpp: Gestreamte Werkzeugaufrufe in `llama-langchain4j` brechen am ersten Fragment ohne Argumente oder ohne Namen ab
- **Zeit:** 2026-10-18 17:00 UTC
- **Datei:** `llama-langchain4j/src/main/java/net/ladenthin/llama/langchain4j/StreamingChunkAssembler.java:131` (`acceptToolCallFragment`), Aufrufer `JllamaStreamingChatModel.doChat`
- **Schwere:** mittel
- **Beschreibung:** Jedes `tool_calls`-Fragment wurde als `PartialToolCall` an den Handler gegeben. langchain4j 1.22.0 baut ihn mit `ensureNotBlank(name)` und `ensureNotEmpty(partialArguments)` (mit `javap -c` am Jar nachgesehen). Upstreams `server_chat_msg_diff_to_json_oaicompat` lässt `arguments` weg, solange der Diff keine hat, und `common_chat_msg_diff::compute_diffs` meldet einen neuen Aufruf mit dem bis dahin geparsten Stand – das erste Fragment trägt typischerweise nur `id` und `name`, und laut Kommentar dort darf der Name anfangs leer sein. Dann warf `PartialToolCall.builder().build()` eine `IllegalArgumentException` aus dem Stream-Callback, `doChat` fing sie und rief `onError`: der ganze gestreamte Werkzeugaufruf schlug fehl. Die vorhandenen Tests schickten nur Fragmente, die schon Argumente trugen.
- **Fix:** Ein Fragment mit leeren Argumenten oder solange der Name fehlt fließt weiter in den zusammengesetzten Aufruf, wird aber nicht als `PartialToolCall` gemeldet (wie langchain4js eigene OpenAI-Anbindung, die nur nicht leere Argumentstücke meldet).
- **Reproduktion:** `StreamingChunkAssemblerTest#toolCallDeltaWithoutArguments_doesNotFailTheStream`, `#toolCallDeltaBeforeItsName_doesNotFailTheStream`
- **Rot-Grün-Nachweis:** ohne Fix beide rot („Unexpected exception thrown: java.lang.IllegalArgumentException: partialArguments cannot be null or empty“ bzw. „… name cannot be null or blank“), mit Fix grün (2/2, skipped 0); Gegenprobe per `git stash` rot (2), nach `pop` grün. Ganzes Modul grün (`StreamingChunkAssemblerTest` 9/9, `LangChain4jMappingTest` 23/23, `JsonSchemaElementSerializerTest` 8/8, `TestModelPathsTest` 8/8; die Integrationstests überspringen sich ohne Modell wie zuvor). Das Modul hat weder Spotless noch SpotBugs.
- **CI-Bedarf:** `JllamaToolCallingIntegrationTest#streamedRequiredToolCallComesBackAsToolExecutionRequest` (Job `test-java-llama-langchain4j-integration`, kein Release-Gate) im nächsten Lauf ansehen: ob er vor dem Fix rot war, hängt davon ab, ob das CI-Modell den Namen allein in einem Fragment sendet.
- **Status:** gefixt
- **Commit:** java-llama.cpp `4d599fa`

<a id="f133"></a>
### F133 – java-llama.cpp: `setPoolingType(UNSPECIFIED)` lässt ein vorher gesetztes `--pooling` stehen
- **Zeit:** 2026-10-18 20:40 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/parameters/ModelParameters.java:652` (`setPoolingType`)
- **Schwere:** niedrig
- **Beschreibung:** Die Javadoc sagt, mit `UNSPECIFIED` werde der Parameter nicht weitergegeben und das Modell nutze seine eigene Pooling-Strategie. Der Code überspringt bei `UNSPECIFIED` nur das Setzen; ein vorher gesetzter Typ bleibt in der Map, `setPoolingType(MEAN).setPoolingType(UNSPECIFIED)` lädt das Modell also mit `--pooling mean`. `setRopeScaling(UNSPECIFIED)` entfernt die Option seit F13 – dessen Eintrag nennt `setPoolingType` als Vorbild, das es nicht war.
- **Fix:** `UNSPECIFIED` ruft `removeOption(ModelOption.POOLING)`, wie `setRopeScaling`; Javadoc um den Satz zum Entfernen ergänzt.
- **Reproduktion:** `ModelParametersTest#testSetPoolingTypeUnspecifiedRemovesAnEarlierPoolingType`
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: not map containing ["--pooling"->ANYTHING] but: was <{--fit=on, --pooling=mean}>“, die übrigen 92 Tests der Klasse grün), mit Fix 93/93 (skipped 0), Gegenprobe per `git stash` rot, danach wieder grün. `LlamaArchitectureTest` 13/13, `WireNameRegistryTest` 8/8, `ModelParametersExtendedTest` 148/148; `spotless:check` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `7231e71`

<a id="f134"></a>
### F134 – java-llama.cpp: fünf `InferenceParameters`-Wither „`null` clears“ behalten den alten Wert
- **Zeit:** 2026-10-18 21:05 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/parameters/JsonParameters.java:245` (`withOptionalJson`), Aufrufer `InferenceParameters.withPrompt`/`withInputPrefix`/`withInputSuffix`/`withGrammar`/`withToolChoice`
- **Schwere:** mittel
- **Beschreibung:** Die fünf Wither dokumentieren `null` als „clears“ (bei `withPrompt`: „clears any prior prompt“). Alle gehen über `withOptionalJson`, das bei `null` nur `this` zurückgibt. Ein früher gesetzter Wert bleibt also im Request: `withGrammar(g).withGrammar(null)` schickt die Grammatik weiter mit, die Generierung bleibt eingeschränkt, ebenso `tool_choice` oder ein Infill-Präfix. Der vorhandene Test `testToJsonStringNull` setzt `null` nur auf ein frisches Objekt und sieht das nicht.
- **Fix:** `withOptionalJson(field, null)` entfernt einen vorhandenen Eintrag (neue Instanz, Original unverändert) und gibt nur dann `this` zurück, wenn der Eintrag fehlt; Javadoc angepasst.
- **Reproduktion:** `InferenceParametersTest#test{Prompt,InputPrefix,InputSuffix,Grammar,ToolChoice}NullClearsAnEarlierValue`, `JsonParametersTest#withOptionalJson_nullRemovesAnEarlierValue`
- **Rot-Grün-Nachweis:** ohne Fix rot – alle fünf Wither-Tests (z. B. „Expected: not map containing ["grammar"->ANYTHING] but: was <{grammar="x"}>“) und der Basistest („expected: not same but was: <TestBuilder{keys=[grammar], values=redacted}>“); mit Fix `InferenceParametersTest` 111/111 und `JsonParametersTest` 26/26 (skipped 0), Gegenprobe per `git stash` wieder rot. PIT auf `JsonParameters`: 18/18 getötet. `LlamaArchitectureTest` 13/13, `WireNameRegistryTest` 8/8, `OpenAiRequestMapperTest` 31/31; `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `f35e7f2`

<a id="f137"></a>
### F137 – java-llama.cpp: `Session.fork` prüft eine negative Ziel-Slot-ID erst nach Sichern und Wiederherstellen
- **Zeit:** 2026-10-18 22:30 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/Session.java:250` (`fork`)
- **Schwere:** niedrig
- **Beschreibung:** `fork(newSlotId, filepath)` prüft vor den Modellaufrufen nur `newSlotId == slotId`. Eine negative ID erreicht `model.saveSlot(slotId, filepath)` (schreibt die Datei) und `model.restoreSlot(newSlotId, filepath)`, das im nativen Teil an der ungültigen ID scheitert – der Aufrufer bekommt eine `LlamaException` statt der `IllegalArgumentException`, die der Konstruktor für genau diesen Wert wirft, und eine geschriebene Scratch-Datei. Die Konstruktorprüfung („Validate here, not per request“) kommt erst danach zum Zug. Die Javadoc nannte als `IllegalArgumentException`-Fall nur die gleiche ID.
- **Fix:** `fork` weist eine negative ID vor jedem Modellaufruf mit derselben Meldung wie der Konstruktor ab; Javadoc ergänzt.
- **Reproduktion:** `SessionTest#forkToANegativeSlotIdRejectedBeforeTheModelIsTouched` (mit `null`-Modell: jeder Modellaufruf zeigt sich als `NullPointerException`)
- **Rot-Grün-Nachweis:** ohne Fix rot („Unexpected exception type thrown, expected: <java.lang.IllegalArgumentException> but was: <java.lang.NullPointerException>“), mit Fix 3/3 (skipped 0), Gegenprobe per `git stash` rot, danach wieder grün. `SessionStateTest` 13/13, `LlamaArchitectureTest` 13/13; `spotless:apply` und `spotbugs:check` grün. Den nativen Pfad (Datei geschrieben, `LlamaException` aus `restoreSlot`) habe ich nicht laufen lassen: keine native Bibliothek und kein Modell in dieser Umgebung.
- **Status:** gefixt
- **Commit:** java-llama.cpp `965b540`

<a id="f138"></a>
### F138 – java-llama.cpp: `OpenAiServerConfig.heartbeatMillis(0)` lässt jeden Streaming-Request als Client-Fehler scheitern
- **Zeit:** 2026-10-18 22:55 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiServerConfig.java` (`Builder.heartbeatMillis`), Verbraucher `OpenAiCompatServer` (vier `scheduleAtFixedRate(…, getHeartbeatMillis(), getHeartbeatMillis(), …)`)
- **Schwere:** niedrig
- **Beschreibung:** Der Builder nimmt jeden Wert. Das Intervall ist die Periode von `scheduleAtFixedRate`, die bei ≤ 0 `IllegalArgumentException` wirft; der Streaming-Handler fängt genau diese Klasse als Fehler der Anfrage. Mit `heartbeatMillis(0)` – naheliegend als „kein Heartbeat“ – antwortet daher jeder Streaming-Request mit HTTP 200 und nur `data: {"error":{"message":"IllegalArgumentException","type":"invalid_request_error",…}}`, ohne `[DONE]` (gemessen mit dem modellfreien Testserver); nicht gestreamte Anfragen gehen. Der Fehler sieht nach einem Client-Fehler aus und nennt die Einstellung nicht.
- **Fix:** `heartbeatMillis` weist einen nicht positiven Wert beim Konfigurieren mit `IllegalArgumentException` ab, Javadoc ergänzt.
- **Reproduktion:** `OpenAiServerConfigTest#heartbeatIntervalOfZeroIsRejected`; das Laufzeitsymptom zeigte eine nicht übernommene Probe in `OpenAiCompatServerHttpTest` (Streaming-Chat gegen `FakeBackend` mit `heartbeatMillis(0)`).
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected java.lang.IllegalArgumentException to be thrown, but nothing was thrown.“), mit Fix 7/7 (skipped 0), Gegenprobe per `git stash` rot, danach wieder grün. `OpenAiCompatServerHttpTest` 54/54, `OpenAiServerCliTest` 45/45, `LlamaArchitectureTest` 13/13; `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `6a13092`

<a id="f143"></a>
### F143 – java-llama.cpp: `RouterClient.awaitModelLoaded(id, 0)` meldet für ein geladenes Modell einen Timeout, ohne abzufragen
- **Zeit:** 2026-10-19 01:05 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/RouterClient.java:210` (`awaitModelLoaded`)
- **Schwere:** niedrig
- **Beschreibung:** Die Schleife `while (System.currentTimeMillis() < deadline)` prüft die Frist vor der ersten Abfrage. Mit `timeoutMillis` 0 (die naheliegende Frage „ist es jetzt geladen?“) oder einer Frist, die vor der ersten Abfrage abläuft, wird `GET /models` nie gefragt: der Aufruf wirft „did not reach LOADED within 0 ms (last status: null)“ für ein geladenes Modell, und auch die dokumentierten Sofort-Fehler (Modell nicht gelistet, Worker gescheitert) treten nie ein. Dieselbe Klasse wie der schon behobene Überlauf bei `Long.MAX_VALUE` (Test `awaitModelLoaded_longMaxValueTimeoutStillPolls`: „would skip polling and report a timeout for a loaded model“).
- **Fix:** Die Methode fragt mindestens einmal ab und prüft die Frist erst nach einer Abfrage, vor der Pause.
- **Reproduktion:** `RouterClientTest#awaitModelLoaded_zeroTimeoutStillPollsOnce`
- **Rot-Grün-Nachweis:** ohne Fix rot („Unexpected exception thrown: java.lang.IllegalStateException: Model 'qwen' did not reach LOADED within 0 ms (last status: null)“), mit Fix 20/20 (skipped 0, die Timeout- und Sofort-Fehler-Tests unverändert grün), Gegenprobe per `git stash` rot, danach wieder grün. `LlamaArchitectureTest` 13/13; `spotless:check` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `058153f`

<a id="f158"></a>
### F158 – java-llama.cpp: `ModelParameters.setPredict(-2)` verspricht „until context filled“, der Server erzeugt damit je Anfrage nur ein Token
- **Zeit:** 2026-10-19 00:10 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/parameters/ModelParameters.java:202` (Javadoc `setPredict`)
- **Schwere:** niedrig bis mittel (still: keine Fehlermeldung, nur abgeschnittene Antworten)
- **Beschreibung:** Gefunden beim Nachprüfen von F69. Upstreams `-n` (`common/arg.cpp:1661`, b11639) nennt „-2 = until context filled“ **nur** für `LLAMA_EXAMPLE_COMPLETION`, für den Server lautet die Beschreibung „-1 = infinity“; der Parser übernimmt jeden Wert. Im Server übernimmt ein Slot `params_base.n_predict`, wenn die Anfrage `-1` sendet oder das Feld weglässt (`server-context.cpp:1964`), und `has_budget()` (`:495`) zählt nur `-1` als unbegrenzt: bei `-2` ist `n_remaining()` negativ, und die Grenzprüfung (`:2040`, `n_gen > 0 && !has_budget()`) beendet die Erzeugung nach dem ersten Token mit `STOP_TYPE_LIMIT`. Aus dem Quelltext abgeleitet, **nicht gemessen**.
- **Fix:** Dokumentation korrigiert (ohne Test, Nutzerregel): Javadoc nennt nur noch `-1 = infinity` und erklärt, dass `-2` eine Einstellung des Completion-Tools ist und hier nach dem ersten Token stoppt. **Offen, Entscheidung des Nutzers:** `-2` in `ModelParameters.setPredict` mit `IllegalArgumentException` abweisen – modellfrei testbar, ändert aber das Verhalten eines öffentlichen Setters.
- **Messung (J1, `BUGHUNT-MEASUREMENTS.md` in `b689d7b`):** bestätigt – `-2`: 1 Token, `stop_type="limit"`; `-1`: 2042 Tokens (bis Kontextende); `16`: 16. Dazu J2: `withNPredict(0)` erzeugt ebenfalls 1 Token, nicht 0.
- **Nachtrag (Entscheidung des Nutzers nach der Messung):** `setPredict` lehnt Werte unter `-1` mit `IllegalArgumentException` ab; Javadoc von `setPredict` und `withNPredict` sagt „`0` verhält sich wie `1`“. Test `ModelParametersExtendedTest#testSetPredictBelowMinusOneIsRejected` (`-2`, `-3`, `Integer.MIN_VALUE`) ersetzt `testSetPredictFillContext`, dazu `testSetPredictZero`.
- **Rot-Grün-Nachweis:** ohne die Prüfung rot (3 von 3), mit ihr grün (`ModelParametersExtendedTest`, `ModelParametersTest`, `WireNameRegistryTest`: 252 Tests).
- **Nachtrag (Branch-Review 2026-10-20):** Die Übersicht nannte bis zum Branch-Review noch „Doku gefixt, Verhalten offen“; das Verhalten ist seit `ae07a53` geändert (siehe oben).
- **Status:** gefixt
- **Commit:** a977a1e (Doku), ae07a53 (Verhalten)

<a id="f165"></a>
### F165 – java-llama.cpp: `OpenAiCompatServer` ohne `-c` bewirbt 8192 + 2048 Tokens statt des Fensters des geladenen Modells
- **Zeit:** 2026-10-19 14:40 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiServerCli.java` (`toServerConfig`), `OpenAiCompatServer.java` (`main`)
- **Schwere:** mittel (ein Client, der seinen Prompt nach `/props`/`/api/show` bemisst, schickt bis zu 5× zu viel Kontext)
- **Historie:** Die Budgets kamen nur aus `-c` (seit F70 als ganzes Slot-Fenster aufgeteilt); ohne `-c` blieben die festen Vorgaben stehen.
- **Messung (J3, `BUGHUNT-MEASUREMENTS.md` in `b689d7b`):** Draft-Modell ohne `-c`: `/props` `n_ctx` 10240, `/api/show` `num_ctx 10240`, `/slots` `n_ctx` 2048 (= `n_ctx_train`).
- **Fix:** Entscheidung des Nutzers: das echte `n_ctx` bewerben. Neue Überladung `toServerConfig(supportsVision, modelFtype, slotContextSize)`; `main` übergibt `n_ctx` des ersten Slots aus `getMetricsTyped()`. Ein bekanntes Slot-Fenster geht vor `-c` (die Last kann kleiner ausfallen, etwa durch `--fit`), sonst gilt wie bisher `-c` bzw. die Vorgabe.
- **Tests:** `OpenAiServerCliTest#loadedSlotContextIsAdvertisedWithoutContextSize`, `#loadedSlotContextWinsOverTheContextSizeArgument`, `#unknownSlotContextFallsBackToTheContextSizeArgument`.
- **Rot-Grün-Nachweis:** mit der alten Auswahl (nur `-c`) rot (2 von 3, der Rückfalltest ist beidseitig grün), mit dem Fix grün (`OpenAiServerCliTest`, `OpenAiCompatServer*Test`: 114 Tests). Die Verdrahtung in `main` braucht ein Modell und ist hier nur übersetzt; Prüfung mit dem Befehl aus J3 (erwartet: 2048 in `/props` und `/api/show`).
- **Status:** gefixt
- **Commit:** 62413a9

<a id="f166"></a>
### F166 – java-llama.cpp: Leere Werkzeug-Argumente im Verlauf lassen die ganze Folgeanfrage mit 400 scheitern
- **Zeit:** 2026-10-19 15:00 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/parameters/ParameterJsonSerializer.java` (`buildMessages(List<ChatMessage>)`)
- **Schwere:** niedrig (mit dem gemessenen Modell nicht erreichbar, im Eintrittsfall aber Totalausfall der Folgeanfrage)
- **Historie:** `buildMessages` schreibt `ToolCall.getArgumentsJson()` seit den Werkzeugaufrufen im Verlauf wörtlich.
- **Messung (J4, `BUGHUNT-MEASUREMENTS.md` in `b689d7b`):** Qwen2.5-1.5B liefert für ein Werkzeug ohne Parameter 20/20 `"{}"`, nie `""`; mit `""` im Verlauf antwortet der Server `InvalidRequestException: Failed to parse tool call arguments as JSON … attempting to parse an empty input`, mit `"{}"` normal.
- **Fix:** Entscheidung des Nutzers: leere (auch nur aus Leerraum bestehende) Argumente als `"{}"` schreiben, alles andere unverändert.
- **Tests:** `ParameterJsonSerializerTest#testBuildMessages_emptyToolCallArguments_areWrittenAsAnEmptyObject` (`""`, `" "`, `"\n"`), `#testBuildMessages_toolCallArguments_arePassedOnVerbatim`.
- **Rot-Grün-Nachweis:** ohne den Fix rot (3 von 3), mit ihm grün (186 Tests in den Serializer-, Multimodal-, `InferenceParameters`- und `ChatRequest`-Klassen).
- **Nicht abgedeckt (offen):** Die Übersetzungen des Servers bauen den Verlauf ohne `buildMessages`: `ResponsesApiSupport.java:218` (`asText("")`) und `AnthropicApiSupport.java:229` (`input` fehlt → `""`) können weiterhin `""` schreiben. **Nachtrag 2026-10-19:** behoben als F172.
- **Status:** gefixt (Java-API-Pfad; Server-Übersetzungen als F172)
- **Commit:** afc40be

<a id="f167"></a>
### F167 – java-llama.cpp: `completeBatch`/`chatBatch` laufen auf dem gemeinsamen `ForkJoinPool` und nutzen auf wenigen Kernen nicht alle Slots
- **Zeit:** 2026-10-19 15:30 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/LlamaModel.java` (`completeBatch`, `completeBatchWithStats`, `chatBatch`), neu `BatchDispatch.java`
- **Schwere:** niedrig bis mittel (Durchsatz; die Javadoc versprach die `setParallel`-Slots)
- **Historie:** Jede Anfrage lief über `CompletableFuture.supplyAsync`, also auf dem gemeinsamen Pool mit `availableProcessors() - 1` Threads, den sie zudem mit der Anwendung teilte.
- **Messung (J5, `BUGHUNT-MEASUREMENTS.md` in `b689d7b`):** 8 Slots, 8 Anfragen: 16 Kerne → 8 gleichzeitig (3225 ms); 4 Kerne → 3 (14067 ms); 16 Kerne mit Pool-Parallelität 3 → 3.
- **Fix:** Entscheidung des Nutzers: `completeBatch`/`completeBatchWithStats` übergeben alle Anfragen im Aufrufer-Thread (`requestCompletion`) und empfangen dann der Reihe nach; scheitert eine Übergabe, werden die bereits übergebenen freigegeben (`releaseTask`). Für `chatBatch` gibt es keine Trennung von Übergabe und Warten (`handleChatCompletions` wartet in einem nativen Aufruf), daher läuft dort jede Anfrage auf einem eigenen Daemon-Thread eines Executors, der genau so groß ist wie der Batch und mit dem Aufruf endet. Beides in `BatchDispatch` (ohne native Bibliothek testbar, `LlamaModel` lädt sie im statischen Initialisierer). Die SpotBugs-Ausnahme für den bewussten Rethrow ist von `LlamaModel` dorthin gezogen, zwei neue mit Begründung (`HES_LOCAL_EXECUTOR_SERVICE`, `ITC_INHERITANCE_TYPE_CHECKING`).
- **Tests:** `BatchDispatchTest` (8): alle Übergaben vor dem ersten Empfang, Übergabe im Aufrufer-Thread, Freigabe nach gescheiterter Übergabe, Empfang aller trotz Fehler, mehr gleichzeitige Anfragen als der gemeinsame Pool Threads hat (Latch über `commonPool().getParallelism() + 2`), Reihenfolge, Fehler nach den anderen, leere Eingabe.
- **Rot-Grün-Nachweis:** mit dem alten Mechanismus (`supplyAsync` je Anfrage, keine Freigabe) rot (4 von 8; die übrigen 4 prüfen unverändertes Verhalten), mit dem Fix grün. Ein modellgestützter Lauf (J5-Probe) steht aus: erwartet `maxProcessingSlots=8` auch auf 4 Kernen für `completeBatch`.
- **Status:** gefixt
- **Commit:** 5b4e409

<a id="f168"></a>
### F168 – java-llama.cpp: `JllamaNative.takeStrings` gibt ein leeres Nicht-NULL-Array nicht frei
- **Zeit:** 2026-10-19 16:30 UTC
- **Datei:** `llama-ffm/src/main/java/net/ladenthin/llama/ffm/JllamaNative.java` (`takeStrings`)
- **Schwere:** niedrig (8 Byte je Aufruf mit leerem Ergebnis, praktisch unerreichbar)
- **Historie:** Die Prüfung `pointer == NULL || count == 0` vor dem `try` stammt aus der ersten Fassung der FFM-Bindungen; `jllama_rpc_server_devices` legt auch für 0 Einträge ein Array an.
- **Messung (J8, `BUGHUNT-MEASUREMENTS.md` in `b689d7b`):** ein absichtlich falsch ausgerichteter Nicht-NULL-Zeiger mit Anzahl 0: `jllama_strings_free` direkt bricht mit `free(): invalid pointer` ab (Exit 134), über `takeStrings` überlebt die JVM – die Freigabe wurde nie gerufen.
- **Fix:** Entscheidung des Nutzers: bei Nicht-NULL immer freigeben. Nur noch `NULL` kehrt früh zurück; die Freigabefunktion ist über eine paketinterne Überladung `takeStrings(out, length, free)` einsetzbar.
- **Tests:** `JllamaNativeTest#takeStringsFreesANonNullArrayEvenWhenItIsEmpty`, `#takeStringsDoesNotFreeANullArray`.
- **Rot-Grün-Nachweis:** lokal mit JDK 25 (Toolchain) und einem selbst gebauten `libjllama.so` (b11639, statisch): mit der alten Prüfung rot (1 von 9, der neue Test), mit dem Fix grün; `mvn -f llama-ffm/pom.xml verify` grün (29 Tests, 16 modellgestützte übersprungen, kein Modell).
- **Status:** gefixt
- **Commit:** 8269149

<a id="f172"></a>
### F172 – java-llama.cpp: Responses- und Anthropic-Übersetzung schreiben leere Tool-Argumente in den Verlauf
- **Zeit:** 2026-10-19 21:30 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/ResponsesApiSupport.java` (`function_call`), `AnthropicApiSupport.java` (`tool_use`)
- **Schwere:** niedrig (laut: die Folgeanfrage scheitert mit 400)
- **Historie:** Rest von F166, dort als „Nicht abgedeckt“ vermerkt: F166 behob den Java-API-Pfad (`buildMessages`), die beiden Übersetzungen bauen den Verlauf selbst.
- **Beschreibung:** Ein `function_call` ohne `arguments` (Responses) und ein `tool_use` ohne `input` (Anthropic, `MissingNode.toString()` ist `""`; `input: null` ergab `"null"`) landeten als `""` bzw. `"null"` in `function.arguments`. Upstream parst jedes Argument des Verlaufs mit `json::parse` und lehnt `""` mit 400 ab (J4: „attempting to parse an empty input“).
- **Fix:** Entscheidung des Nutzers: dieselbe Regel wie F166 – leere oder nur aus Leerraum bestehende Argumente (Responses) bzw. fehlender oder `null`-`input` (Anthropic) werden `"{}"`.
- **Tests:** `ResponsesApiSupportTest#functionCallWithoutArgumentsBecomesAnEmptyArgumentsObject` (fehlend, `""`, Leerraum), `AnthropicApiSupportTest#toolUseWithoutInputBecomesAnEmptyArgumentsObject` (fehlend, `null`).
- **Rot-Grün-Nachweis:** ohne den Fix rot (beide, `was ""`), mit ihm grün (75 Tests der beiden Klassen).
- **Status:** gefixt
- **Commit:** java-llama.cpp `f1bf93b`

<a id="f173"></a>
### F173 – java-llama.cpp: Responses-Übersetzung verwirft `refusal`-Teile von Assistenten-Nachrichten im Verlauf
- **Zeit:** 2026-10-19 21:45 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/ResponsesApiSupport.java` (`inputContentText`, `inputContentWithImages`)
- **Schwere:** niedrig (still: der Verlauf verliert, dass das Modell abgelehnt hat)
- **Historie:** Offener Verdacht vom 2026-10-18 (TODO (11)). Die Übersetzung las Textteile nur über `text`; ein `refusal`-Teil trägt seinen Text unter `refusal`.
- **Beschreibung:** Ein Responses-Client schickt die eigene frühere Antwort mit `{"type":"refusal","refusal":"…"}` zurück. Der Teil fiel weg, eine Nachricht nur aus einer Ablehnung wurde leer. Upstream übersetzt ihn zu einem Teil vom Typ `refusal`, den der eigene Chat-Parser mit 400 ablehnt (`common/chat.cpp`, nur Text und Medien).
- **Fix:** Entscheidung des Nutzers: als Text übernehmen. Neuer Helfer `partText` (Text aus `text`, bei Typ `refusal` aus `refusal`), genutzt von beiden Lesewegen; neben einem Bild wird er ein `text`-Teil.
- **Tests:** `ResponsesApiSupportTest#refusalPartsOfTheHistoryAreKeptAsText` (Text + Ablehnung, nur Ablehnung), `#refusalPartsNextToAnImageAreKeptAsTextParts`.
- **Rot-Grün-Nachweis:** ohne den Fix rot (2 von 54: `was "a"`, `was ""`), mit ihm grün (54 von 54).
- **Status:** gefixt
- **Commit:** java-llama.cpp `0fd166f`

<a id="f174"></a>
### F174 – java-llama.cpp: `/props` meldet `total_slots` fest als 1
- **Zeit:** 2026-10-19 22:00 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiSseFormatter.java` (`propsJson`), `OpenAiServerConfig`, `OpenAiServerCli` (`toServerConfig`), `OpenAiCompatServer` (`main`, `handleProps`)
- **Schwere:** niedrig (still falsch; ein Client, der Anfragen auf die Slots verteilt, sieht einen Slot)
- **Historie:** Offener Verdacht vom 2026-10-18 (TODO (9)): `propsJson` setzte `total_slots` seit seiner Einführung als Konstante; upstreams `/props` meldet `params.n_parallel` (`get_res_props`).
- **Beschreibung:** Mit `--parallel 4` – oder ohne die Option, wenn llama.cpp die Slotzahl selbst wählt – meldete `/props` einen Slot. Die Zahl kannte nur die CLI, nicht `OpenAiServerConfig`.
- **Fix:** Entscheidung des Nutzers: `OpenAiServerConfig.totalSlots` (Builder, mindestens 1, Vorgabe 1) wird in `/props` gemeldet. Der Starter nimmt die Slotzahl des geladenen Modells (`getMetricsTyped().getSlotMetrics().size()`, wie schon das Kontextfenster je Slot, F70/J3); ohne sie entscheidet `--parallel`, ohne das 1. Neue Überladung `toServerConfig(vision, ftype, slotContext, totalSlots)`; die Metriken werden in `main` nur noch einmal abgefragt.
- **Tests:** `OpenAiCompatServerHttpTest#propsEndpointReportsTheSlotCountOfTheConfiguration` (über HTTP), `OpenAiSseFormatterTest#propsJsonReportsContextLengthAndModalities` (ergänzt), `OpenAiServerConfigTest#totalSlotsDefaultsToOneAndKeepsTheConfiguredCount`, `#totalSlotsBelowOneIsRejected`, `OpenAiServerCliTest#slotCountOfTheLoadedModelIsAdvertised`, `#unknownSlotCountFallsBackToTheParallelArgumentAndThenToOne`.
- **Rot-Grün-Nachweis:** mit einem Platzhalter, der das alte Verhalten behält (Getter liefert 1, `propsJson` ignoriert den Parameter), rot (5 Tests), mit dem Fix grün (129 Tests der vier Klassen); SpotBugs grün. Der Starter selbst (`main`) braucht ein Modell und ist nur übersetzt.
- **Status:** gefixt
- **Commit:** java-llama.cpp `569415d`

<a id="f175"></a>
### F175 – java-llama.cpp: `RouterClient.findModel`/`awaitModelLoaded` erkennen einen Alias nicht
- **Zeit:** 2026-10-19 22:20 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/RouterClient.java` (`findModel`), `value/RouterModel.java`, `json/RouterModelsResponseParser.java`
- **Schwere:** niedrig (laut, aber falsch: Timeout bzw. „not listed“, obwohl der Router das Modell bedient)
- **Historie:** Offener Verdacht vom 2026-10-17 (TODO (4)). Upstream löst Modellnamen über Aliase auf (`--alias` im Preset, `server_models::get_meta`) und listet sie in `GET /models` unter `aliases` (`server-models.cpp:2141`, b11639); `RouterModel` las das Feld nie.
- **Beschreibung:** `awaitModelLoaded("<alias>")` meldete nach der vollen Wartezeit „Router does not list model“, `findModel("<alias>")` lieferte nichts – während eine Anfrage unter demselben Namen bedient wurde.
- **Fix:** Entscheidung des Nutzers (API-Erweiterung): `RouterModel.getAliases()` und `isNamed(name)` (Id oder Alias, groß/klein genau wie upstream), ein neuer Konstruktor mit Aliasen (die bisherigen delegieren mit leerer Liste), der Parser liest `aliases`, `findModel` vergleicht über `isNamed`. README-Hinweis.
- **Tests:** `RouterClientTest#findModel_matchesByAlias` (Stub-Router, auch `awaitModelLoaded` über den Alias), `RouterModelsResponseParserTest#parsesAliasesAndDefaultsThemToEmpty`, `RouterModelTest#aliasesDefaultToEmptyAndAreKeptUnmodifiable`, `#isNamed_matchesTheIdAndEveryAliasAndNothingElse`.
- **Rot-Grün-Nachweis:** mit Platzhaltern (`getAliases()` leer, `isNamed` nur Id) rot (4 Tests), mit dem Fix grün (54 Tests, 2 modellgestützte übersprungen); PIT auf `RouterModel` 18/18 getötet; SpotBugs grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `11c7477`

<a id="f176"></a>
### F176 – java-llama.cpp: `rerank(String, String...)` liefert die Dokumente in Hash-Reihenfolge
- **Zeit:** 2026-10-19 22:35 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/LlamaModel.java` (`rerank(String, String...)`), `json/RerankResponseParser.java` (`toDocumentScores`)
- **Schwere:** niedrig (still: Reihenfolge zufällig; gleiche Dokumente teilen einen Eintrag)
- **Historie:** Rest von F167 (TODO (15)): F167 behob den Batch-Teil, die Map-Form blieb. Sie sammelte seit jeher in einer `HashMap`, nach Dokumenttext.
- **Beschreibung:** Die Ergebnisse kommen in Dokumentreihenfolge aus dem Server; die `HashMap` gab sie in Hash-Reihenfolge zurück (gemessen: `zeta, alpha, mid, b, a` → `mid` an zweiter Stelle). Gleiche Dokumente fallen auf einen Eintrag zusammen – ohne Datenverlust, weil gleicher Text gleich bewertet wird, aber undokumentiert.
- **Fix:** Entscheidung des Nutzers: `LinkedHashMap` in Ergebnisreihenfolge, herausgelöst als `RerankResponseParser.toDocumentScores` (ohne native Bibliothek testbar); Javadoc von `rerank(String, String...)` nennt das Zusammenfallen und verweist für eine Liste auf `rerank(boolean, …)`. Die Signatur bleibt (`LlamaOutput` nimmt eine Map).
- **Tests:** `RerankResponseParserTest#toDocumentScores_keepsTheOrderOfTheDocuments`, `#toDocumentScores_equalDocumentsShareOneEntryAtTheirFirstPosition`.
- **Rot-Grün-Nachweis:** mit der `HashMap` rot (1 von 14, `item 1: was "mid"`), mit der `LinkedHashMap` grün (14 von 14); PIT auf `RerankResponseParser` 6/6 getötet; SpotBugs grün (Parameter `Collection`, vorbelegte Größe).
- **Status:** gefixt
- **Commit:** java-llama.cpp `3299e6a`

<a id="f177"></a>
### F177 – java-llama.cpp: `/v1/rerank`: `top_n: 0` liefert alle Ergebnisse, ein negatives `top_n` wird durchgereicht, `usage` fehlt
- **Zeit:** 2026-10-19 22:55 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OaiRerankSupport.java`, `LlamaModelBackend.java` (`rerank`), `src/main/cpp/json_helpers.hpp` (`rerank_results_to_json`)
- **Schwere:** niedrig (still abweichend von upstream)
- **Historie:** Offener Verdacht vom 2026-10-18 (TODO (8)). `OaiRerankSupport` behandelte `top_n <= 0` seit seiner Einführung als „alle“. Upstreams `format_response_rerank` (`server-common.cpp:1535`, b11639) behält `min(top_n, n)` Einträge – bei 0 keinen, bei einem negativen Wert ein `resize` auf eine riesige Größe – und setzt immer `usage` aus der Summe von `tokens_evaluated`. Das native Array (`rerank_results_to_json`) reichte `tokens_evaluated` nicht weiter.
- **Beschreibung:** `top_n: 0` lieferte alle Dokumente statt keines, `top_n: -1` ebenfalls alle (upstream scheitert intern), und `usage` fehlte in der OpenAI-förmigen Antwort ganz.
- **Fix:** Entscheidung des Nutzers: `top_n: 0` behält keinen Eintrag (OpenAI- und TEI-Form); ein negatives `top_n` ist eine ungültige Anfrage (400), geprüft in `LlamaModelBackend.rerank` vor dem nativen Aufruf; `rerank_results_to_json` reicht `tokens_evaluated` weiter (0, wenn das Ergebnis es nicht trägt), und `toOaiResponse` setzt `usage.prompt_tokens`/`total_tokens` auf die Summe. Javadoc von `handleRerank` nennt das neue Feld.
- **Tests:** C++: `RerankResultsToJson.TokensEvaluatedArePassedOnAndDefaultToZero`. Java: `OaiRerankSupportTest#negativeTopNIsAnInvalidRequest`, `#topNZeroReturnsNoResultsLikeUpstream`, `#toOaiResponseCarriesTheUsageOfTheEvaluatedTokens`, `#readTopNReturnsValueOrNegativeOne` (um 0 ergänzt).
- **Rot-Grün-Nachweis:** C++ ohne den Fix rot (`Which is: -1`), mit ihm grün (9/9 der Suite); Java ohne den Fix rot (3 von 16), mit ihm grün (16/16, mit `OpenAiCompatServerHttpTest` 71); SpotBugs grün. Ende zu Ende mit einem Reranker (CI: `OpenAiServerRerankIntegrationTest`) nicht lokal gelaufen.
- **Status:** gefixt
- **Commit:** java-llama.cpp `232985f`

<a id="f178"></a>
### F178 – java-llama.cpp: Ollama-Routen melden `done_reason` falsch und `/api/generate` keine Tokenzahlen
- **Zeit:** 2026-10-19 23:20 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OllamaApiSupport.java`, `OpenAiCompatServer.java` (`streamOllamaChat`, `handleOllamaGenerate`)
- **Schwere:** niedrig (still: ein Client erkennt abgeschnittene Antworten im Stream und bei `/api/generate` nie)
- **Historie:** Offener Verdacht vom 2026-10-17 (TODO (2)); die Tokenzahlen der gestreamten `/api/chat`-Zeile kamen schon mit F98. Ollama meldet `done_reason` `"length"` für eine durch `num_predict` abgeschnittene Antwort und sonst `"stop"`, auch bei Werkzeugaufrufen.
- **Beschreibung:** Blockierendes `/api/chat` reichte OpenAIs `finish_reason` durch, also auch `"tool_calls"`, das Ollama nicht kennt; gestreamtes `/api/chat` und beide `/api/generate`-Formen meldeten immer `"stop"`. `/api/generate` meldete außerdem weder `prompt_eval_count` noch `eval_count`, obwohl die Antwort des Backends sie trägt.
- **Fix:** Entscheidung des Nutzers: eine Abbildung `toOllamaDoneReason` (`length` → `length`, alles andere → `stop`) für alle vier Wege. Der Stream merkt sich den `finish_reason` des Endchunks (`finishReasonOf`) für die Abschlusszeile; `/api/generate` baut aus der Antwort des Backends ein `Generated` (Text, Grund, Zählungen): aus der OpenAI-Completion `finish_reason` und `usage`, aus der nativen Infill-Antwort `stop_type` (`limit` → `length`) und `tokens_evaluated`/`tokens_predicted`. Fehlen die Zählungen, bleiben die Felder weg. `extractCompletionText`/`extractInfillContent` entfielen (nur noch von Tests benutzt).
- **Tests:** `OpenAiCompatServerHttpTest#ollamaChatDoneReasonIsLengthWhenCutOffAndStopForToolCalls` (blockierend und gestreamt, `length`/`tool_calls`/`stop`), `#ollamaGenerateDoneReasonAndCountsFollowTheCompletion` (blockierend und gestreamt), `#ollamaGenerateWithSuffixReadsTheInfillStopType`; `OllamaApiSupportTest#generateTextExtractors` (umgestellt), `#generateWithoutCountsOmitsThemAndToolCallsAreStop`.
- **Rot-Grün-Nachweis:** ohne den Fix rot (3 HTTP-Tests, `was "stop"`), mit ihm grün (94 Tests der beiden Klassen; Pakete `server`, `json`, `value` 417 grün); SpotBugs grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `6eb8081`

<a id="f179"></a>
### F179 – java-llama.cpp: Ollama `/api/generate` liest weder `raw` noch `system` und vervollständigt immer roh
- **Zeit:** 2026-10-19 23:45 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OllamaApiSupport.java` (`isRaw`, `toOpenAiChatRequestFromGenerate`, `Generated.fromChat`), `OpenAiCompatServer.java` (`handleOllamaGenerate`)
- **Schwere:** niedrig (still: ein Instruct-Modell setzt den Text fort statt zu antworten, `system` und Bilder gingen verloren)
- **Historie:** Offener Verdacht vom 2026-10-18 (TODO (16)). Die Route war seit ihrer Einführung „prompt completion / FIM“. Ollama wendet ohne `"raw": true` die Chat-Vorlage des Modells an, mit `system` als Systemnachricht; Continue (`Ollama.ts`) setzt `raw` nur, wenn der Aufrufer es verlangt.
- **Beschreibung:** Jede Anfrage ohne `suffix` ging als roher Prompt an `/v1/completions`; `system`, `images`, `format` und `think` wurden ignoriert.
- **Fix:** Entscheidung des Nutzers: Ollamas Semantik. Mit `suffix` wie bisher Infill, mit `raw: true` wie bisher die rohe Completion (`system` ignoriert, wie bei Ollama), sonst der Chat-Weg: aus der Anfrage wird eine `/api/chat`-Anfrage (`system` + Nutzer-`prompt` mit `images`, dazu `options`, `format`, `think`) und über die vorhandene Übersetzung geschickt; die Antwort trägt `response`, `thinking` (aus `reasoning_content`), `done_reason` und die Zählungen. README und `package-info` beschreiben die drei Wege; der modellgestützte `OpenAiServerCompletionIntegrationTest` bekam eine `raw`-Variante.
- **Tests:** `OpenAiCompatServerHttpTest#ollamaGenerateWithoutRawAppliesTheChatTemplateLikeOllama` (Aufzeichnungs-Backend: Chat-Weg, `system`, Bild, `thinking`, Zählung), `#ollamaGenerateWithRawCompletesTheBarePromptAndIgnoresSystem`; `OpenAiServerCompletionIntegrationTest#ollamaGenerateRawRoundTrip` (Modell, CI).
- **Rot-Grün-Nachweis:** ohne den Fix rot (`was "completions"`), mit ihm grün (96 Tests); der F178-Test gab nun explizit `raw: true` an, weil er die Completion-Abbildung prüft; SpotBugs grün. Die modellgestützte Variante läuft nur in CI bzw. beim lokalen Agenten.
- **Status:** gefixt
- **Commit:** java-llama.cpp `75b3767`

<a id="f180"></a>
### F180 – java-llama.cpp: OpenAI-Mapper verwerfen `logit_bias`, `n_probs` und `n` still
- **Zeit:** 2026-10-20 00:10 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiRequestMapper.java` (`applyCommonFields`, `toCompletionParameters`), `parameters/InferenceParameters.java` (`withLogitBiasJson`)
- **Schwere:** mittel (still: ein gesperrtes Token wurde trotzdem erzeugt, `n: 3` bekam eine Antwort, Wahrscheinlichkeiten fehlten)
- **Historie:** Offener Verdacht vom 2026-10-18 (TODO (10)), teilweise durch F95/F123 behoben (Sampling-Felder, `logprobs`/`top_logprobs` des Chat-Wegs). Upstreams Schema liest `logit_bias` als Paar-Array oder als OpenAI-Objekt (`server-schema.cpp:382`, b11639), `n_probs` direkt und `n` als Alias von `n_cmpl` (`:62`).
- **Beschreibung:** Chat (blockierend und gestreamt, und damit die Anthropic-, Responses- und Ollama-Übersetzungen) und das gestreamte `/v1/completions` ignorierten `logit_bias`, `n_probs` und `n`. Ein früherer Test (`unknownFieldsIgnored`) schrieb das Ignorieren von `"n":3` sogar fest. Das gestreamte `/v1/completions` baut seine Chunks aus dem erzeugten Text und kann keine Wahrscheinlichkeiten tragen.
- **Fix:** Entscheidung des Nutzers: `logit_bias` (Objekt oder Array) wird unverändert über die neue öffentliche `InferenceParameters.withLogitBiasJson` weitergereicht (ein JSON-Wert, wie die anderen Fragment-Setter geprüft), `n_probs` über `withNProbs`; `n > 1` ist eine ungültige Anfrage (400), weil diese Wege eine Antwort liefern; auf dem gestreamten `/v1/completions` sind `logprobs`/`n_probs` eine ungültige Anfrage mit dem Hinweis auf `stream: false`, bis (14) den Stream nativ macht.
- **Tests:** `OpenAiRequestMapperTest#logitBiasAndNProbsAreForwardedOnBothPaths`, `#moreThanOneChoiceIsRefusedBecauseOneAnswerComesBack`, `#probabilitiesOnTheStreamedCompletionAreRefused`, `#unknownFieldsIgnored` (ohne `n`); `InferenceParametersTest#withLogitBiasJson_storesTheValueAndRefusesASecondOne`.
- **Rot-Grün-Nachweis:** mit Platzhalter-Setter und unverändertem Mapper rot (4 Tests), mit dem Fix grün (327 Tests der Mapper-, Parameter-, Registry- und Server-Klassen); SpotBugs grün.
- **Status:** gefixt (der Rest unter (14), Wahrscheinlichkeiten im gestreamten `/v1/completions`, mit F186)
- **Commit:** java-llama.cpp `3d001d2`

<a id="f181"></a>
### F181 – java-llama.cpp: JNI-Schalen geben Fehlermeldungen als Modified UTF-8 zurück
- **Zeit:** 2026-10-20 01:10 UTC
- **Datei:** `llama/src/main/cpp/jni_utf8.hpp` (neu), `jllama.cpp` (`throw_api_status`, Modell-Laden), `jni_helpers.hpp` (`report_cxx_exception_impl`), `native_server.cpp`, `rpc_bridge.cpp`, `train_engine.cpp`
- **Schwere:** niedrig bis mittel (still verfälschte Meldung auf HotSpot, Prozessabbruch unter Android-CheckJNI)
- **Historie:** Offener Verdacht vom 2026-10-18 (Musterdurchsicht nach F71, TODO (6)). Gemessen auf HotSpot 21 mit einer eigenen JNI-Sonde: `GetStringUTFChars` liefert für U+1F600 `ED A0 BD ED B8 80` (CESU-8), `NewStringUTF` macht aus `F0 9F 98 80` ein falsches Zeichen und verschluckt Folgebytes.
- **Beschreibung:** `ThrowNew` und `NewStringUTF` lesen Modified UTF-8. Die Meldungen aus llama.cpp zitieren Pfade und Anfragetext in Standard-UTF-8. Die Trainer-Meldung für `missing-😀.gguf` kam als `missing-`, U+00F0 U+009F U+0098 U+0080 und `.g` an (der Rest des Pfads fehlte). Die Eingabeseite des Trainers (`GetStringUTFChars`) ist dagegen nicht erreichbar: Jackson schreibt ein Zeichen der Zusatzebenen als maskiertes Surrogatpaar, die Konfiguration ist also dort ASCII, wo sich die Kodierungen unterscheiden (gemessen: `"Hallo \\ud83d\\ude00 Welt"`). Die Fehlermeldungen von Anfragen an das Modell zitieren in den gemessenen Fällen keinen Anfragetext (Probe mit einem Zufallsmodell).
- **Fix:** Neuer Header `jni_utf8.hpp` (nur `jni.h`, keine gecachten IDs, deshalb auch in `train_engine.cpp` nutzbar): `throw_utf8_impl` und `new_string_utf8_impl`. ASCII bleibt bei `ThrowNew`/`NewStringUTF`, alles andere geht als Bytes durch `String(byte[], "UTF-8")`, und Javas Dekoder ersetzt ungültige Folgen durch U+FFFD. Umgestellt sind alle Stellen mit fremdem Text: `throw_api_status`, das Modell-Laden, `report_cxx_exception_impl` (die JNI-Ausnahmegrenze), `NativeServer`, `RpcServer` und der Rückgabetext von `LlamaTrainer`. Stellen mit literalem ASCII-Text bleiben, wie sie sind. Die Eingabeseite des Trainers bekommt einen Kommentar, warum sie sicher ist.
- **Tests:** C++ `JniUtf8.*` (6 in `test_jni_helpers.cpp`: ASCII bleibt bei `ThrowNew`, 4-Byte-Zeichen und ungültiges Byte erreichen den Konstruktor als Originalbytes, dasselbe für `new_string_utf8_impl` und für `jni_guard_impl`); Java `LlamaTrainerTest#aFourByteCharacterInAPathComesBackIntactInTheErrorMessage` (modellfrei, braucht nur `libjllama`; überspringt, wo die Dateinamen-Kodierung den Pfad nicht darstellen kann) und `#aFourByteCharacterInTheTrainingTextReachesTheTrainer` (hält die Jackson-Maskierung fest, von der die Eingabeseite abhängt).
- **Rot-Grün-Nachweis:** C++: mit durchreichenden Platzhaltern 4 von 6 rot, mit dem Fix 6 von 6 grün, die ganze Suite 640/640. Java gegen die alte Bibliothek rot (`but: was "…/missing-ð.g"`), gegen die neu gebaute grün; `LlamaLoggerTest`, `NativeLibraryLoadSmokeTest`, `RpcServerTest` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `6f530e9`

<a id="f182"></a>
### F182 – java-llama.cpp: Gestreamte Anfragen antworten mit 200, bevor das Backend läuft
- **Zeit:** 2026-10-20 01:50 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiCompatServer.java` (`ResponseStream`, die fünf Streaming-Methoden); `llama-atmosphere-agent` (Tests, README)
- **Schwere:** mittel (eine abgelehnte Anfrage kam als `200` an; ein Client, der nur den Status prüft, sah eine leere Antwort, der Atmosphere-Agent beendete den Zug still)
- **Historie:** Offener Verdacht vom 2026-10-17 (TODO (3)), damals als Designabweichung zur Entscheidung gestellt und durch `nativeRejectionOfAStreamedChatCompletionIsAnInvalidRequestEvent` festgeschrieben. Upstream (`server-context.cpp:4906`, b11639) holt vor dem Streamen das erste Ergebnis und antwortet bei einem Fehler mit dessen Status.
- **Beschreibung:** Jede der fünf Streaming-Routen (`/v1/chat/completions`, `/v1/completions`, `/api/chat`, `/v1/messages`, `/v1/responses`) schickte `200` und die Stream-Header vor dem Aufruf des Backends; Anthropic und Responses schrieben zusätzlich sofort ihre Eröffnungsereignisse. Eine sofort abgelehnte Anfrage (`InvalidRequestException`, `IllegalArgumentException`) oder ein sofortiger Fehler wurde dadurch zum Fehlerereignis unter `200`.
- **Fix:** `ResponseStream` beginnt die Antwort erst mit dem ersten Schreiben: Status `200`, Header, dann der mit `startWith` gesetzte Vorspann (die Eröffnungsereignisse von Anthropic und Responses), dann der Text. Ein Heartbeat ist ebenfalls ein Schreiben, denn er hält einen Client über einen langen Prompt hinweg; danach bleibt ein Fehler ein Ereignis. Vorher beantwortet `answerIfNotStarted` die Anfrage mit `400`/`500` und dem Fehlerformat der Route (OpenAI, Ollama `{"error"}`, Anthropic `invalid_request_error`/`api_error`), wie ohne `stream`. Nach einer solchen Antwort sind weitere Schreibversuche wirkungslos (ein Heartbeat zwischen Antwort und Abbruch schreibt nichts).
- **Tests:** `OpenAiCompatServerHttpTest#aRejectionBeforeTheFirstChunkIsA400OnEveryStreamingRoute`, `#aFailureBeforeTheFirstChunkIsA500OnEveryStreamingRoute`, `#aRejectionAfterTheFirstChunkIsAnInvalidRequestEvent`, `#aRejectionAfterAHeartbeatIsAnEvent`; die vier bestehenden Ereignis-Tests (Responses `response.failed`, Anthropic-Fehlertypen) laufen jetzt mit einem Backend, das nach einem Chunk scheitert (`ScriptedStreamBackend`). Im Agenten `AtmosphereWireContractTest#engineFailureBeforeTheFirstChunkFailsTheTurn` (neu) und `#midStreamEngineFailureCompletesSilentlyRatherThanErroring` (jetzt wirklich nach einem Chunk).
- **Rot-Grün-Nachweis:** vor dem Umbau rot (die beiden Vor-dem-ersten-Chunk-Tests), danach 63/63 in `OpenAiCompatServerHttpTest` und alle Server-Testklassen grün; der Agent gegen den installierten neuen Kern: vorher rot (`API returned 500: model exploded` statt `null`), nach der Aufteilung `verify` grün (278 Tests, 63 übersprungen, modell- oder bildschirmgebunden); SpotBugs grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `395d9d1`

<a id="f183"></a>
### F183 – java-llama.cpp: Anthropic `stop_sequence` ist immer `null`
- **Zeit:** 2026-10-20 02:25 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/AnthropicApiSupport.java` (`toOpenAiChatRequest`, `toAnthropicResponse`, `stopSequenceOf`, `messageDeltaEvent`), `AnthropicStreamTranslator.java`, `OpenAiRequestMapper.java`, `parameters/InferenceParameters.java` (`withVerbose`), `parameters/RequestField.java` (`VERBOSE`)
- **Schwere:** niedrig (ein Client mit mehreren `stop_sequences` erfuhr nicht, welche gegriffen hat, und konnte einen Stopp durch das Wort nicht von einem Ende des Modells unterscheiden)
- **Historie:** Offener Verdacht vom 2026-10-18 (TODO (12)). Upstream setzt `stop_sequence` in `to_json_anthropic` und `to_json_anthropic_stream` aus `stopping_word` (`server-task.cpp:778`/`:959`, b11639); `stop_reason` bleibt `end_turn`. Das Feld `verbose` liest upstreams Schema (`server-schema.cpp:17`), `__verbose` enthält dann `to_json_non_oaicompat()` mit `stopping_word` (blockierend am Objekt, gestreamt am ersten Chunk des Endergebnisses).
- **Beschreibung:** `/v1/messages` übersetzt aus der OpenAI-Antwort, und die trägt das Stoppwort nicht; `stop_sequence` war deshalb fest `null`, blockierend wie gestreamt.
- **Fix:** Die übersetzte Anfrage verlangt `verbose: true`; `stopSequenceOf` liest `__verbose.stopping_word` (leer = `null`), `toAnthropicResponse` und der Stream-Übersetzer setzen es als `stop_sequence`, der `message_delta` trägt es. `__verbose` erreicht den Client nicht, weil beide Antworten neu aufgebaut werden. Dafür gibt es `RequestField.VERBOSE` (Vertrag `SCHEMA`, vom C++-Vertragstest gegen upstreams Schema geprüft) und `InferenceParameters.withVerbose`; der OpenAI-Mapper reicht ein `verbose` des Clients weiter.
- **Tests:** `AnthropicApiSupportTest#theTranslatedRequestAsksForTheStopWord`, `#theMatchedStopWordIsTheStopSequence`, `#noMatchedStopWordIsANullStopSequence`; `AnthropicStreamTranslatorTest#theMatchedStopWordIsTheStopSequenceOfTheMessageDelta`, `#withoutAStopWordTheStopSequenceIsNull`; `OpenAiRequestMapperTest#verboseIsForwarded`; `InferenceParametersTest` (`withVerbose`); C++ `test_wire_contracts.cpp` (`verbose` im Schema, 5/5).
- **Rot-Grün-Nachweis:** mit wirkungslosem Platzhalter-Setter rot (5 Tests, die beiden Null-Fälle waren schon grün), mit dem Fix grün (251 Tests der berührten Klassen, `OpenAiCompatServerHttpTest` und `WireNameRegistryTest` eingeschlossen); SpotBugs grün. Zusätzlich Ende-zu-Ende mit einem Zufallsmodell über den echten Server: ohne Stoppwort `max_tokens`/`null`, mit einem Stück der eigenen Greedy-Ausgabe als Stoppwort blockierend und gestreamt `end_turn` mit `stop_sequence: "j%j"`, kein `__verbose` in der Antwort.
- **Status:** gefixt
- **Commit:** java-llama.cpp `a575714`

<a id="f184"></a>
### F184 – java-llama.cpp: Mehrere Prompts in einer Anfrage: alle außer dem ersten still verworfen
- **Zeit:** 2026-10-20 02:50 UTC
- **Datei:** `llama/src/main/cpp/jllama_engine.cpp` (`populate_completion_task`)
- **Schwere:** mittel (still: zwei Prompts, eine Antwort; ein Client, der nach `index` zuordnet, wartet auf eine Antwort, die nie kommt)
- **Historie:** Gefunden bei der Arbeit an TODO (14): `populate_completion_task` übernimmt nur `tokenized_prompts[0]`, und alle drei Wege, die es nutzen (blockierendes `/v1/completions` über `completions_oai`, `handleCompletions`, der Roh-Stream), posten genau eine Aufgabe. Upstream (`handle_completions_impl`, b11639) legt pro Prompt eine Aufgabe an und antwortet mit einer Wahl je Prompt.
- **Beschreibung:** `{"prompt": ["Hi", "Ho"]}` lieferte `200` mit einer einzigen Wahl für `Hi`; `Ho` verschwand ohne Hinweis. Eine Token-Liste (`[72, 105]`) ist dagegen ein einziger Prompt.
- **Fix:** Mehr als ein Prompt ist eine ungültige Anfrage (`400`, `a request takes one prompt, this one has N prompts; send them as separate requests`). Mehrere Prompts wirklich zu bedienen hieße mehrere Aufgaben je Anfrage und, gestreamt, eine Zusammenführung nach `index` – für keinen Weg hier vorhanden, und keiner der Clients, für die der Server gebaut ist, schickt das.
- **Tests:** C++ `CApi.SeveralPromptsInOneRequestAreRefusedNotDropped` (`test_c_api.cpp`, Zufallsmodell): zwei Prompts auf allen drei Wegen abgelehnt, mit der Anzahl in der Meldung; eine Token-Liste und ein Array mit einem Prompt angenommen.
- **Rot-Grün-Nachweis:** vor dem Fix rot (alle drei Wege `JLLAMA_STATUS_OK`), danach grün; ganze C++-Suite 641/641.
- **Status:** gefixt
- **Commit:** java-llama.cpp `c4062bd`

<a id="f185"></a>
### F185 – java-llama.cpp: OpenAI-Antworten ohne `id` und ohne `model`
- **Zeit:** 2026-10-20 03:20 UTC
- **Datei:** `llama/src/main/cpp/jllama_engine.cpp` (`name_completion_task`, aufgerufen von `blocking_completion` und `streaming_request`); `args/ModelOption.java` (`ALIAS`), `parameters/ModelParameters.java` (`setAlias`), `server/OpenAiServerCli.java` (`toModelParameters`)
- **Schwere:** niedrig bis mittel (still: ein Client, der Antworten oder Stream-Chunks nach `id` zuordnet oder das Modell aus `model` liest, bekam leere Zeichenketten)
- **Historie:** Gefunden bei der Arbeit an TODO (14), als die OAI-Antworten mit einem Zufallsmodell angesehen wurden. Upstream setzt in `handle_completions_impl` (`server-context.cpp:4884`, b11639) `oaicompat_cmpl_id = gen_chatcmplid()` und `oaicompat_model = meta->model_name` für jede Aufgabe; die Engine setzte keins von beiden, seit sie die Aufgaben selbst baut.
- **Beschreibung:** `/v1/chat/completions`, `/v1/completions` und jeder Stream-Chunk (also auch die Übersetzungen) trugen `"id":""` und `"model":""`. Nach dem Engine-Fix hätte `OpenAiCompatServer --model-id x` außerdem in `/v1/models` `x` gemeldet, in den Antworten aber den Dateinamen, weil die CLI die Id nicht an llama.cpp weitergab.
- **Fix:** `name_completion_task` setzt beide Felder wie upstream (eine `chatcmpl-`-Id je Aufgabe, also je Stream eine). Neu `ModelOption.ALIAS` (`--alias`, Vertrag `SERVER_PARSER`, vom C++-Vertragstest gegen upstreams Parser geprüft) und `ModelParameters.setAlias`; die CLI reicht `--model-id` als Alias weiter.
- **Tests:** C++ `CApi.OpenAiAnswersCarryAnIdAndTheModelName` (`test_c_api.cpp`, Zufallsmodell: blockierende Completion und Chat mit `chatcmpl-`-Id und dem Modellnamen aus den Meta-Daten, verschiedene Ids, ein Stream mit genau einer Id über alle Chunks); Java `ModelParametersTest#testSetAlias`, `OpenAiServerCliTest#anExplicitModelIdBecomesTheModelsAlias`; `test_model_flags.cpp` (`--alias` im Server-Parser).
- **Rot-Grün-Nachweis:** C++ vor dem Fix rot (`id` und `model` leer), danach grün, Suite 642/642; Java mit wirkungslosem Platzhalter rot (2), danach grün (304 Tests der Parameter-, CLI- und Registry-Klassen). Ende-zu-Ende über den echten Server mit `--model-id my-model`: `/v1/models` und die Chat-Antwort nennen beide `my-model`.
- **Status:** gefixt
- **Commit:** java-llama.cpp `0be9b33`

<a id="f186"></a>
### F186 – java-llama.cpp: Das gestreamte `/v1/completions` ist ein selbstgebauter Stream ohne `usage`, Wahrscheinlichkeiten und Token-Prompt
- **Zeit:** 2026-10-20 04:00 UTC
- **Datei:** `llama/src/main/cpp/jllama_engine.{hpp,cpp}` (`completions_oai_stream_request`), `jllama_api.{h,cpp}` (`jllama_completions_oai_stream_request`), `jllama.cpp` + `jllama.h` (`requestCompletionOaiStream`), `LlamaModel.java` (`streamCompletionsOai`), `server/LlamaModelBackend.java` (`streamCompletions`), `server/OpenAiRequestMapper.java` und `OpenAiSseFormatter.java` (toter Completion-Weg entfernt); `llama-ffm` (`JllamaNative`, `FfmLlamaModel.completionsOaiStream`); `docs/c-api.md`, README
- **Schwere:** mittel (dieselbe Anfrage bedeutete gestreamt etwas anderes als blockierend: ohne `usage`, ohne `logprobs`, Token-Liste abgelehnt)
- **Historie:** Offener Verdacht vom 2026-10-18 (TODO (14)); F180 hatte `logprobs`/`n_probs` auf diesem Weg ausdrücklich abgelehnt, bis der Stream nativ ist. Upstream legt `usage` auf den letzten Chunk (`to_json_oaicompat`, `server-task.cpp:391`, b11639) und streamt `logprobs` je Chunk (`:1068`).
- **Beschreibung:** `LlamaModelBackend.streamCompletions` lief über `LlamaModel.generate` und baute `text_completion`-Chunks aus dem erzeugten Text: der letzte trug kein `usage`, Wahrscheinlichkeiten fehlten, und `OpenAiRequestMapper.toCompletionParameters` lehnte einen `prompt` als Array ab, den das blockierende `/v1/completions` (wörtlich an upstream) annimmt.
- **Fix:** Neuer nativer Einstieg, der den Körper wie das blockierende `completions_oai` mit `oaicompat_completion_params_parse` liest und als `TASK_RESPONSE_TYPE_OAI_CMPL` streamt; der Empfang ist der vorhandene Hüllen-Empfang des Chat-Streams. `LlamaModelBackend.streamCompletions` reicht den Körper (mit `stream: true`) wörtlich weiter. `toCompletionParameters` samt seiner Ablehnungslisten und `OpenAiSseFormatter.completionChunk` hatten danach keinen Aufrufer mehr und sind entfernt; die Mapper-Tests, die gemeinsame Felder prüften, laufen jetzt über den Chat-Weg. Mehrere Prompts lehnt der native Weg seit F184 ab, gestreamt wie blockierend.
- **Tests:** Java `OpenAiServerCompletionIntegrationTest#aStreamedCompletionIsUpstreamsStream` (modellgestützt: Token-Prompt, `logprobs`, eine Id, `usage` mit `prompt_tokens` 2, `finish_reason`, `[DONE]`); C++ `CApi.OaiCompletionsStreamDeliversTextCompletionChunksEndingWithUsage` (Zufallsmodell, dazu die Ablehnungen ohne Prompt, mit zwei Prompts und mit Null-Argument); FFM `FfmLlamaModelIntegrationTest#oaiCompletionStreamEndsWithUsage`; `OpenAiRequestMapperTest` umgestellt (33).
- **Rot-Grün-Nachweis:** Der Java-Integrationstest mit einem Zufallsmodell vor dem Umbau rot (`400` für den Token-Prompt), danach grün; C++ 643/643; `llama-ffm verify` grün (17 + 4 + 9) gegen die neu gebaute Bibliothek; `OpenAiCompatServerHttpTest` 63, `NativeLibraryLoadSmokeTest` 6 grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `d98a6ad`

<a id="f187"></a>
### F187 – java-llama.cpp: `addLoraScaledAdapter`/`addControlVectorScaled` senden `FNAME,SCALE`, upstream liest `FNAME:SCALE`
- **Zeit:** 2026-10-19 23:20 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/parameters/ModelParameters.java` (`addLoraScaledAdapter`, `addControlVectorScaled`, neu `scaledField`), `llama/src/test/java/net/ladenthin/llama/parameters/ModelParametersTest.java`
- **Schwere:** mittel (das Modell lädt überhaupt nicht; die Ausnahme nennt die Ursache nicht)
- **Historie:** Gefunden beim Messauftrag `BUGHUNT-MEASUREMENTS.md`, M6: der erste Versuch, die Referenz mit Adapter über `addLoraScaledAdapter(ADAPTER, 1.0f)` zu laden, scheiterte mit `LlamaException: Failed to parse model parameters`, worauf die Messung auf `--lora` auswich. Upstream hat das Format mit `487674fbb` (b7448, „common: fix --override-kv to support comma-separated values“, #18056, 2025-12-17) umgestellt und mit `07fbe19f1` (b7648, #18643) um CSV-Maskierung erweitert; das Komma ist seither der **Listen**trenner, der Feldtrenner ist ein Doppelpunkt. Der Pin ist b11639, die Methode war also rund 4200 Builds tot.
- **Beschreibung:** Beide Methoden rendern `putString(…, fname + "," + scale)`. `common/arg.cpp` am Pin liest den Wert als `FNAME:SCALE,...`: `parse_csv_row(value)`, dann je Element `string_split(item, ':')` mit der Bedingung `parts.size() != 2 → throw`. `--lora-scaled <pfad>,1.0` scheitert daher im argv-Parser mit `invalid stof argument`, und der Aufrufer sieht nur „Failed to parse model parameters“. Gemessen gegen den echten Parser (je ein Kind-JVM über `ServerLauncher` → `NativeServer`, der llama-server-argv wörtlich weitergibt): Form `<abs>,1.0` → `invalid stof argument`, Server startet nicht; `<abs>:1.0` → `lora-scaled format: FNAME:SCALE`, Server startet nicht; `<rel>:1.0` → angenommen, Server lauscht; `--lora <abs>` → angenommen. Abgedeckt war die Methode nur von `ModelParametersTest#testAddLoraScaledAdapter`, das genau das defekte Format festschrieb (`is("adapter.bin,0.5")`) — die Klasse Fehler, die CLAUDE.md für die Wire-Name-Registries beschreibt: ein Test über die String-Abbildung bleibt grün, während das Flag tot ist. `test_model_flags.cpp` sieht es nicht, es hält den **Namen** gegen die Optionstabelle des Servers, nicht das **Werteformat**.
- **Begleitende upstream-Grenze, nicht downstream behebbar:** `string_split` trennt an *jedem* Doppelpunkt (`common/common.h`), also ergibt `X:\models\a.gguf:1.0` drei Teile statt zwei. **Über `--lora-scaled` lässt sich kein absoluter Windows-Pfad ausdrücken**, auch im korrekten Format nicht (Form 2 oben). Ein reiner `,`→`:`-Tausch hätte auf Windows weiter nicht geladen.
- **Fix:** Die beiden Methoden rendern `FNAME:SCALE` über das neue private `scaledField`, das einen Pfad mit Doppelpunkt mit `IllegalArgumentException` ablehnt — die Meldung nennt das Flag, das Format, den Pfad und die zwei Auswege (relativer Pfad, oder ungeskaliert laden und die Skala nach dem Laden mit `LlamaModel.setLoraAdapter` setzen). Eine laute Ablehnung beim Bauen ist besser als ein Ladefehler ohne Ursache; die Grenze ist im Javadoc von `addLoraScaledAdapter` erklärt, auf das `addControlVectorScaled` verweist. Beide gehen jetzt durch `appendCsvValue` (siehe F188).
- **Tests:** `ModelParametersTest#testAddLoraScaledAdapter` und `#testAddControlVectorScaled` auf `FNAME:SCALE` umgestellt; neu `#aScaledPathContainingAColonIsRefusedWithTheReason` (beide Optionen, Meldung nennt Flag, Format und Pfad, relativer Pfad als Gegenprobe im selben Test).
- **Rot-Grün-Nachweis:** Vorher scheiterte `new LlamaModel(base().addLoraScaledAdapter(adapter, 1f))` mit `Failed to parse model parameters`. Nachher, end-to-end mit dem Adapterpaar aus M6 (`ggml-org/stories15M_MOE`, Wegwerf-Probe, danach entfernt): `--lora-scaled <rel>:1.0` lädt und liefert denselben Text wie die `--lora`-Referenz, `:0.0` denselben wie ohne Adapter, und der Server meldet `"scale":0.0` zurück — die Skala wirkt also, was über `ModelParameters` vorher gar nicht möglich war. Danach alle Module grün: Kern 2154/0/0/18 (2136 ausgeführt), `ctest` 643/643, SpotBugs 0, Agent 281/0/0/60, `llama-ffm` 30/0/0/0, `llama-langchain4j` 55/0/0/7, `llama-kotlin` 6/0/0/0.
- **Status:** gefixt
- **Commit:** java-llama.cpp `15f857a`

<a id="f188"></a>
### F188 – java-llama.cpp: Die vier „can be repeated“-Adapteroptionen ersetzen den vorigen Wert statt ihn zu ergänzen
- **Zeit:** 2026-10-19 23:20 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/parameters/CliParameters.java` (neu `appendCsvValue`), `ModelParameters.java` (`addLoraAdapter`, `addLoraScaledAdapter`, `addControlVector`, `addControlVectorScaled`)
- **Schwere:** niedrig bis mittel (von mehreren Adaptern wirkt nur der letzte, ohne Hinweis)
- **Historie:** Gefunden beim Beheben von F187 — dieselben vier Methoden, dieselbe Wire-Option, aber ein eigener Fehler.
- **Beschreibung:** Alle vier versprechen im Javadoc „can be repeated to use multiple adapters“ bzw. „can be repeated to add multiple control vectors“. Der Parametersatz ist aber eine `LinkedHashMap`, die nach dem Wire-String schlüsselt, und `putString` ist `parameters.put(option, value)` — ein zweiter Aufruf überschreibt den ersten still. Für einen Setter ist das richtig, für ein `addX` nicht. Upstream nimmt genau diese Optionen seit #18056/#18643 als **einen** komma-separierten Wert, Ergänzen ist also das, was sie zusammensetzt.
- **Fix:** Neues `protected final appendCsvValue(ModelOption, String)` in `CliParameters`: hängt an einen vorhandenen Wert mit Komma an und schreibt jeden Wert als CSV-Feld — in Anführungszeichen mit verdoppelten inneren Anführungszeichen, sobald er ein Komma oder ein Anführungszeichen trägt. Das ist nötig, weil upstreams `parse_csv_row` ein öffnendes Anführungszeichen nur am **Feldanfang** erkennt, ein Pfad mit Komma also anders nicht übergeben werden kann. Die vier Methoden nutzen es; `setOverrideKv` bleibt unverändert, es ist ein Setter, der den Wert des Aufrufers durchgibt.
- **Tests:** `ModelParametersTest#theRepeatableAdapterOptionsAccumulateInsteadOfReplacing` (alle vier Optionen, je zwei Aufrufe, mit den erwarteten Listenwerten einschließlich `c.gguf:0.5,d.gguf:1.0`) und `#aValueCarryingACommaOrAQuoteIsWrittenAsAQuotedCsvField` (Komma, Anführungszeichen, und ein einfacher Wert vor einem zu maskierenden).
- **Rot-Grün-Nachweis:** Die Akkumulationszusicherungen sind gegen `putString` rot (der erste Wert fehlt). Die End-to-end-Probe aus F187 zeigt zwei skalierte Adapter als einen Wert: `--lora-scaled ..\models\moe_shakespeare15M.gguf:0.5,second.gguf:0.25`. Modullaufzahlen wie in F187.
- **Status:** gefixt
- **Commit:** java-llama.cpp `15f857a`

<a id="f189"></a>
### F189 – java-llama.cpp: Ohne `--model-id` nennen `/v1/models` und die Antworten dasselbe Modell verschieden
- **Zeit:** 2026-10-19 23:20 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiServerCli.java` (`toModelParameters`), `OpenAiServerConfig.java` (Javadoc von `getModelId`), `llama/src/test/java/net/ladenthin/llama/server/OpenAiServerCliTest.java`
- **Schwere:** niedrig bis mittel (ein Client, der den aus `/v1/models` gelesenen Namen gegen das `model`-Feld der Antwort hält, sieht zwei verschiedene; zusätzlich steht ein absoluter lokaler Pfad in jeder Antwort, auch über das Netz)
- **Historie:** Gefunden beim Messauftrag `BUGHUNT-MEASUREMENTS.md`, M5(b). F185 hatte die Hälfte erwischt: sein Test hieß `anExplicitModelIdBecomesTheModelsAlias` und sein Kommentar beschrieb genau diesen Fehler für den **expliziten** Fall, schrieb für den Vorgabefall aber fest, dass *kein* `--alias` gesetzt wird.
- **Beschreibung:** `OpenAiServerCli.ParsedArgs.getModelId()` leitet ohne `--model-id` den Dateinamen ab, und `/v1/models` meldet ihn. Das `model`-Feld einer Antwort kommt dagegen nicht aus der Konfiguration, sondern aus llama.cpp, und `toModelParameters()` setzte `--alias` nur unter `if (modelId != null)`. Ohne Alias ist upstreams Kette für den Modellnamen (`tools/server/server-context.cpp:1531`) Alias → `common_params_model::get_name()` → Dateiname, und `get_name()` (`common/common.h:312`) gibt für eine lokale Datei `path` zurück — der Dateiname-Zweig, den upstreams eigener Kommentar „fallback: derive model name from file name“ beschreibt, ist für `-m <pfad>` also unerreichbar. Gemessen mit zwei Kind-JVMs: ohne `--model-id` meldete `/v1/models` `AMD-Llama-135m-code.Q2_K.gguf`, jede Antwort und jeder Stream-Chunk den vollen Pfad; mit `--model-id my-model` stimmten beide; `NativeServer --alias my-model` als Gegenprobe ebenfalls. Die `id`-Hälfte von F185 ist nicht betroffen (`chatcmpl-`, eine Id pro Stream, in allen drei Fällen).
- **Fix:** `toModelParameters()` setzt den Alias **immer**, aus `getModelId()` statt aus `modelId`, damit der Vorgabewert dieselbe Zusage trägt wie das Flag — die Zusage, die der Kommentar an der Stelle schon aussprach („this keeps them equal to /v1/models“).
- **Bewusst nicht geändert:** der eingebettete Weg. `OpenAiCompatServer(LlamaModel, OpenAiServerConfig)` bekommt ein schon geladenes Modell und kann kein `--alias` mehr setzen, also trägt eine Antwort dort weiter den Namen, mit dem das Modell geladen wurde (beobachtet in M2, wo die Probe `modelId("m")` setzt und die Antworten den Pfad nennen). Das steht jetzt im Javadoc von `OpenAiServerConfig.getModelId()` mit dem Hinweis, dass ein einbettender Aufrufer sein Modell mit `ModelParameters.setAlias(…)` laden muss. Das `model`-Feld einer Antwort nachträglich zu überschreiben wäre die Alternative; sie würde über die Engine etwas behaupten, was diese nicht getan hat, und `NativeServer` im Attach-Modus könnte es ohnehin nicht nachvollziehen.
- **Tests:** `OpenAiServerCliTest#theAdvertisedModelIdBecomesTheModelsAliasWithAndWithoutTheFlag` — ersetzt `anExplicitModelIdBecomesTheModelsAlias`, prüft beide Fälle und dass der Alias und `getModelId()` derselbe String sind.
- **Rot-Grün-Nachweis:** Der alte Test schrieb das Gegenteil fest und ist gegen den Fix rot. End-to-end (Wegwerf-Probe, danach entfernt), Server ohne `--model-id` über `ServerLauncher`: `/v1/models`, die blockierende Antwort und jeder Stream-Chunk nennen alle `AMD-Llama-135m-code.Q2_K.gguf`, kein absoluter Pfad erreicht eine Antwort, `id` ist `chatcmpl-…` und je Stream eine. Modullaufzahlen wie in F187.
- **Status:** gefixt
- **Commit:** java-llama.cpp `15f857a`

<a id="f190"></a>
### F190 – java-llama.cpp: Das Skript-Backend des Agenten schreibt unmaskierte Steuerzeichen in seine JSON-Chunks (Modul baut auf Windows nicht grün)
- **Zeit:** 2026-10-19 22:35 UTC
- **Datei:** `llama-atmosphere-agent/src/test/java/net/ladenthin/llama/atmosphere/ScriptedBackend.java` (`quote`), neu `ScriptedBackendTest.java`
- **Schwere:** niedrig (reiner Testcode, nichts davon wird ausgeliefert — aber das Agentenmodul ist auf Windows nicht baubar)
- **Historie:** Gefunden beim Messauftrag `BUGHUNT-MEASUREMENTS.md`, M8: der modellfreie Lauf des Agentenmoduls war rot, deterministisch in 3 von 3 Läufen. Die drei Agenten-Jobs laufen nur auf Linux, wo `System.lineSeparator()` ein einzelnes LF ist, darum hat CI es nie gesehen.
- **Beschreibung:** `quote` maskierte nur `"`, `\` und `\n` und schrieb alles andere unverändert durch. RFC 8259 verbietet U+0000..U+001F unmaskiert in einem String, also war der Chunk kein JSON, sobald der Text ein Steuerzeichen trug. Gemessen mit beiden `quote`-Fassungen gegen Jackson: **31 von 32** Steuerzeichen ergaben ungültiges JSON, nur `\n` war behandelt; der Wert selbst wurde in keiner Fassung verändert, es ging rein um Gültigkeit. `AgentSessionTest#aLoopRunsItsStepsThroughTheFrontEndUntilTheMarker` ist der einzige Test, der `System.lineSeparator()` einspeist: auf Windows landete ein rohes CR im JSON-String (`Illegal unquoted character ((CTRL-CHAR, code 13))`), der Client verwarf den Strom (`HTTP/1.1 header parser received no bytes`), jeder Loop-Schritt bekam nichts, und der Stillstandswächter beendete den Loop nach drei Schritten. Was rot wurde, war also eine Zusicherung über die **Antwort**, mehrere Schichten von der Ursache entfernt. `TaskLoop.isComplete` ist für beide Zeilenenden `true`, die Sentinel-Erkennung ist nicht beteiligt.
- **Fix:** `quote` bekommt `\r`, `\t`, `\b`, `\f` und — das Tragende — einen `default`-Zweig, der alles unter `0x20` als `\uXXXX` schreibt, damit ein nicht aufgelistetes Steuerzeichen die Lücke nicht wieder aufreißen kann. Der Test selbst bleibt unangetastet: `System.lineSeparator()` dort ist richtig, weil genau das CRLF überhaupt erst ausübt. Ein `grep` über alle handgerollten JSON-Quote-Helfer im Repository zeigt, dass es diese eine Stelle war.
- **Tests:** Neu `ScriptedBackendTest` (3): jeder Chunk eines Text-Zuges parst als JSON (mit `System.lineSeparator()`, CRLF, LF und einzelnem CR); **jedes** der 32 Steuerzeichen einzeln, also der `default`-Zweig und nicht die vier benannten Fälle; und der Text kommt unverändert aus dem Chunk zurück.
- **Rot-Grün-Nachweis:** Mit der alten `quote`-Fassung kurz zurückgesetzt: `ScriptedBackendTest` 2 Failures + 1 Error (`System.lineSeparator(): chunk is not JSON`, `U+0000: chunk is not JSON`, `JsonParse Illegal unquoted character ((CTRL-CHAR, code 13))`) und `AgentSessionTest` 1 Failure. Mit dem Fix (Datei danach byte-identisch wiederhergestellt) beide dreimal hintereinander grün, und das ganze Modul 281/0/0/60 — die 60 sind die dokumentierten JLine-Fälle. `spotless:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `3b9d96b`

<a id="f197"></a>
### F197 – java-llama.cpp: `n` > 1 wird auf jedem nativen Completion-Weg still auf eine Antwort reduziert (das gestreamte `/v1/completions` verlor dabei den Schutz aus F180)
- **Zeit:** 2026-10-20 10:30 UTC
- **Datei:** `llama/src/main/cpp/jllama_engine.cpp` (`populate_completion_task`), `llama/src/test/cpp/test_c_api.cpp`, `llama/src/test/java/net/ladenthin/llama/server/OpenAiServerCompletionIntegrationTest.java`
- **Schwere:** mittel (zwei Antworten verlangt, eine geliefert, ohne Hinweis — dieselbe Klasse wie F184)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20. F180 hatte `n > 1` im Java-Mapper für den gestreamten `/v1/completions` abgelehnt; F186 (`d98a6ad`) verlegte diesen Weg auf den nativen OAI-Pfad und entfernte `toCompletionParameters` und damit die Ablehnung samt ihrer Testzusicherung.
- **Beschreibung:** Upstreams Schema liest `n` als `n_cmpl` mit den harten Grenzen 1 … `n_parallel` (`server-schema.cpp:62`) und beantwortet die weiteren Completions in Kind-Tasks, die der Server anlegt (`server-context.cpp:4888`). Die Engine stellt genau einen Task ein und legt keine Kinder an. Jedes `n` zwischen 2 und der Slot-Zahl wurde also angenommen und mit einer Wahl beantwortet — gemessen vom Reviewer mit dem Zufallsmodell und `setParallel(2)`: `handleCompletionsOai` mit `n:2` eine Wahl, `streamCompletionsOai` ebenso; erst `n:3` scheiterte am Schema. Betroffen: C-API, FFM, `LlamaModel.handle*`, das blockierende `/v1/completions` (leitet den Körper wörtlich weiter) und seit F186 das gestreamte.
- **Fix:** `populate_completion_task` lehnt `task.params.n_cmpl > 1` nach dem Schema mit `jllama_invalid_request` ab ("a request takes one completion, this one asks for 'n' = N; send separate requests"), wie mehrere Prompts (F184). Das deckt die blockierenden, gestreamten und Chat-Wege an einer Stelle.
- **Tests:** `CApi.SeveralCompletionsInOneRequestAreRefusedNotDropped` (eigenes Modell mit `-np 2`; `jllama_completions_oai`, `jllama_completions` mit `n_cmpl`, der Stream-Request und `jllama_chat_completions` werden abgelehnt, `n:1` liefert eine Wahl), `OpenAiServerCompletionIntegrationTest#aCompletionAskingForSeveralChoicesIsAnswered400OnBothRoutes` (beide `/v1/completions`-Routen, das Testmodell jetzt mit `setParallel(2)`), `#severalChoicesAreRefusedOnTheModelsOwnCompletionAndChatPaths`.
- **Rot-Grün-Nachweis:** Der C-API-Test gegen den alten Code rot (alle vier Wege `OK` statt `INVALID_REQUEST`), mit dem Fix grün; `ctest` grün. Die beiden Java-Tests lokal mit dem Zufallsmodell grün (die zwei Infill-Tests der Klasse scheitern dort erwartungsgemäß, das Zufallsmodell hat keine FIM-Tokens); in CI mit dem echten Modell.
- **Status:** gefixt
- **Commit:** java-llama.cpp `447de71`

<a id="f198"></a>
### F198 – java-llama.cpp: `FfmTextToSpeech.close()` gibt die Engine frei, während `synthesize()` auf einem anderen Thread in ihr läuft (FFM-Hälfte von F16)
- **Zeit:** 2026-10-20 11:00 UTC
- **Datei:** `llama-ffm/src/main/java/net/ladenthin/llama/ffm/FfmTextToSpeech.java`, neu `llama-ffm/src/test/java/net/ladenthin/llama/ffm/FfmTextToSpeechTest.java`
- **Schwere:** mittel (Use-after-free → JVM-Absturz, nur bei `close()` parallel zu einer laufenden Synthese)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20: F16 nennt die Datei als betroffen, auf dem Branch war sie aber unverändert, und F16 stand auf „gefixt“.
- **Beschreibung:** `synthesize()` prüfte `closed.get()` und rief dann ohne Lock `jllama_tts_synthesize`; `close()` setzte das Flag per CAS und rief `jllama_tts_free` → `engine_free` → `delete engine`. Die native Seite zählt keine Nutzer (`jllama_api.cpp`), also löschte ein `close()` von einem anderen Thread die Engine samt ihrem `synthesize_mutex` unter einer laufenden Synthese — dieselbe Prüfen-dann-Rufen-Lücke wie F59 bei `FfmNativeServer`.
- **Fix:** Ein `ReentrantReadWriteLock` wie im JNI-`TextToSpeech`: Synthesen halten das Lese-Lock (die Engine serialisiert sie selbst), `close()` nimmt das Schreib-Lock und wartet so auf jede laufende Synthese; das `closed`-Flag steht unter dem Lock. Die zwei nativen Aufrufe liegen hinter einem `Backend`-Seam, wie bei `FfmNativeServer`, damit die Sperre ohne Bibliothek testbar ist.
- **Tests:** `FfmTextToSpeechTest` (5, ohne Bibliothek und Modell): `closeWaitsForASynthesisInsideTheNativeCall`, `synthesizeAfterCloseIsRefusedWithoutTouchingTheHandle`, `closeTwiceFreesOnce`, `synthesizeReturnsTheBackendsWav`, `synthesizeWithoutTextIsRefused`.
- **Rot-Grün-Nachweis:** Mit dem Seam, aber ohne Lock: `closeWaitsForASynthesisInsideTheNativeCall` rot (`close()` kehrte zurück, während die Synthese noch lief); mit dem Lock grün. `llama-ffm` `verify` 35/35 mit dem Zufallsmodell.
- **Status:** gefixt
- **Commit:** java-llama.cpp `d6be2c5`

<a id="f199"></a>
### F199 – java-llama.cpp: Ein Heartbeat vor dem Eröffnungsevent verschluckt `message_start` (Anthropic) bzw. `response.created` (Responses)
- **Zeit:** 2026-10-20 11:30 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiCompatServer.java` (`ResponseStream`, `streamAnthropic`, `streamResponses`), neu `llama/src/test/java/net/ladenthin/llama/server/ResponseStreamTest.java`
- **Schwere:** mittel (Client erhält einen Stream ohne Eröffnungsevent; Anthropic-SDKs verwerfen ihn)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20.
- **Beschreibung:** Beide Routen planten den Heartbeat-Timer **vor** `out.startWith(translator.begin())`. Feuerte der Heartbeat dazwischen (oder kam die Antwort erst nach einem Heartbeat in Gang), startete er den Stream mit leerem Prelude; `startWith` setzte danach nur noch das Feld, das `body()` nur beim Start schreibt — das Eröffnungsevent ging still verloren.
- **Fix:** `startWith` schreibt den Text sofort, wenn der Stream schon läuft (ein Heartbeat ist ein SSE-Kommentar, kein Event, also bleibt das Eröffnungsevent das erste Event); beide Routen setzen die Eröffnung zudem vor dem Planen des Heartbeats.
- **Tests:** `ResponseStreamTest` (4, über einen `HttpExchange`-Fake ohne Socket): `theOpeningIsWrittenFirstWhenItIsSetBeforeTheFirstWrite`, `anOpeningSetAfterAHeartbeatStartedTheStreamIsStillWritten`, `anErrorBeforeTheFirstWriteIsAnsweredWithItsStatusAndNoOpening`, `anErrorAfterTheFirstWriteIsLeftToTheStream`.
- **Rot-Grün-Nachweis:** Ohne den Fix `anOpeningSetAfterAHeartbeatStartedTheStreamIsStillWritten` rot (erwartet `: ping\nOPEN\nDATA\n`, war `: ping\nDATA\n`); mit ihm grün. `OpenAiCompatServerHttpTest` 63/63.
- **Status:** gefixt
- **Commit:** java-llama.cpp `bbdafda`

<a id="f200"></a>
### F200 – java-llama.cpp: `chatBatch` startet einen Thread pro Anfrage ohne Obergrenze, und ein `Error` beendet das Warten vor den anderen Anfragen
- **Zeit:** 2026-10-20 11:45 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/BatchDispatch.java`, `llama/src/main/java/net/ladenthin/llama/LlamaModel.java` (`chatBatch`), `llama/src/test/java/net/ladenthin/llama/BatchDispatchTest.java`
- **Schwere:** niedrig bis mittel (eine große Liste wird zu ebenso vielen Threads; nach einem `Error` kehrt der Aufruf zurück, während Anfragen noch laufen)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20; `runEachOnItsOwnThread` entstand mit dem Fix, der `chatBatch` vom gemeinsamen `ForkJoinPool` löste.
- **Beschreibung:** `Executors.newFixedThreadPool(requests.size(), ...)`: zehntausend Anfragen waren zehntausend Threads, obwohl der native Scheduler nur so viele gleichzeitig rechnet, wie er Slots hat. Und `asRuntimeException` warf einen `Error` aus der ersten Future, die ihn trug, sofort — die übrigen wurden nicht mehr abgewartet, entgegen der Zusage, dass ein Fehler erst nach allen Anfragen propagiert.
- **Fix:** `runEachOnItsOwnThread(requests, maxThreads, call)`: der Pool hat `min(requests.size(), maxThreads)` Threads (`chatBatch` übergibt `MAX_CHAT_THREADS` = 64), weitere Anfragen starten, sobald ein Thread frei wird; eine Grenze unter 1 wird abgelehnt. Der erste Fehler jeder Art wird gemerkt und erst nach allen Futures geworfen.
- **Tests:** `BatchDispatchTest`: neu `runEachOnItsOwnThread_runsNoMoreRequestsAtOnceThanTheBound` (12 Anfragen, Grenze 3, höchstens 3 gleichzeitig), `runEachOnItsOwnThread_anErrorIsThrownAfterTheOthersFinished`, `runEachOnItsOwnThread_aBoundBelowOneIsRefused`.
- **Rot-Grün-Nachweis:** Mit der neuen Signatur, aber ohne Grenze und ohne den Error-Pfad: die drei neuen Tests rot (12 gleichzeitig statt 3; zwei statt null fertige Anfragen fehlten beim Wurf; keine Ablehnung von 0); mit dem Fix 11/11 grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `3e996db`

<a id="f201"></a>
### F201 – java-llama.cpp: `--model-id` mit Komma (oder abgeleitet aus einem Dateinamen mit Komma) wird zu mehreren llama.cpp-Aliasen; `/v1/models` und die Antworten nennen das Modell verschieden
- **Zeit:** 2026-10-20 12:00 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiServerCli.java`, `llama/src/test/java/net/ladenthin/llama/server/OpenAiServerCliTest.java`
- **Schwere:** niedrig (falscher Modellname in den Antworten; Clients, die den Namen aus `/v1/models` vergleichen, verwerfen sie)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20; Folge von F185, das die ID immer als `--alias` weitergibt.
- **Beschreibung:** Upstreams `--alias` (`common/arg.cpp`, b11639) teilt den Wert an Kommas, strippt jeden Teil und legt ihn in ein `std::set`; die Antworten nennen den ersten Eintrag. `--model-id my,model` meldete in `/v1/models` `my,model`, in jeder Antwort `model`; dasselbe für eine aus `a,b.gguf` abgeleitete ID, und eine ID aus Leerzeichen wurde ganz verworfen (die Antworten nannten wieder den Pfad).
- **Fix:** `--model-id`/`--model-alias` lehnen ein Komma, einen leeren Wert und Leerraum am Rand mit einer Nutzungsmeldung ab; die aus dem Dateinamen abgeleitete ID ersetzt Kommas durch `_` und wird getrimmt (leer → Default-ID), also genau die Form, die der Alias behält.
- **Tests:** `OpenAiServerCliTest`: `aModelIdWithACommaIsRefused`, `aModelIdLlamaCppWouldStripIsRefused`, `aFileNameWithACommaIsAdvertisedAsItsAlias`.
- **Rot-Grün-Nachweis:** Ohne den Fix alle drei rot (keine Ablehnung; abgeleitete ID `a,b.gguf`); mit ihm 54/54 grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `707f0f2`

<a id="f202"></a>
### F202 – java-llama.cpp: Die Anthropic-Route holt für das Stoppwort upstreams ganzes Debug-Objekt samt gerendertem Prompt durch JNI
- **Zeit:** 2026-10-20 12:20 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/AnthropicApiSupport.java`, `OpenAiRequestMapper.java`, `parameters/InferenceParameters.java` (neu `withResponseFields`), `parameters/RequestField.java` (`RESPONSE_FIELDS`), `parameters/ParameterJsonSerializer.java`
- **Schwere:** niedrig (Aufwand: jede Anthropic-Antwort trug den vollständigen Prompt, die Token-Liste und die Generierungseinstellungen zusätzlich, um sie danach zu verwerfen)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20; Folge von F183, das `verbose` setzt, um `stop_sequence` zu füllen.
- **Beschreibung:** `verbose: true` hängt an die OpenAI-Antwort `__verbose` = `to_json_non_oaicompat()` des Endergebnisses: Inhalt, Tokens, `prompt` (der ganze gerenderte Prompt), `generation_settings`, `timings` … Gebraucht wird nur `stopping_word`. Upstreams `response_fields` projiziert genau dieses Objekt (`json_get_nested_values`), war aber weder ein `RequestField` noch vom Mapper durchgereicht.
- **Fix:** `RequestField.RESPONSE_FIELDS` (SCHEMA-Vertrag, gegen `make_llama_cmpl_schema` geprüft) mit `InferenceParameters.withResponseFields(String...)`; der Mapper reicht `response_fields` durch und lehnt alles außer einem String-Array mit 400 ab; die Anthropic-Übersetzung setzt `response_fields: ["stopping_word"]`.
- **Tests:** `AnthropicApiSupportTest.theTranslatedRequestAsksForTheStopWord` (erweitert), `OpenAiRequestMapperTest.responseFieldsAreForwarded`, `responseFieldsThatAreNotStringsAreRefused`, `InferenceParametersTest.withResponseFieldsWritesTheProjectionAsAStringArray`, modellgestützt `OpenAiCompatServerIntegrationTest.theDebugObjectTheAnthropicRouteAsksForCarriesOnlyTheStopWord` (`__verbose` hat genau den Schlüssel `stopping_word`); C++ `test_wire_contracts` prüft den neuen Schlüssel gegen upstreams Schema.
- **Rot-Grün-Nachweis:** Mit `RequestField`/`withResponseFields`, aber dem alten Mapper und der alten Übersetzung: drei Tests rot; mit dem Fix grün (221 Tests der fünf Klassen, Integrationstest 13/13 mit dem Zufallsmodell, ctest Wire 6/6).
- **Status:** gefixt
- **Commit:** java-llama.cpp `05f356b`

<a id="f203"></a>
### F203 – java-llama.cpp: `chat_template_kwargs`: ein Wert mit mehr als einem JSON-Wert wird still auf den ersten gekürzt
- **Zeit:** 2026-10-20 12:35 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/parameters/ParameterJsonSerializer.java` (`buildRawValueObject`; betrifft `InferenceParameters.withChatTemplateKwargs` und `ModelParameters.setChatTemplateKwargs`)
- **Schwere:** niedrig (falscher Template-Parameter ohne Meldung)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20; dieselbe Klasse wie die `FAIL_ON_TRAILING_TOKENS`-Fixes in `JsonParameters` und `ChatRequest`, hier übersehen.
- **Beschreibung:** `OBJECT_MAPPER.readTree(value)` liest den ersten JSON-Wert und ignoriert den Rest: `"true false"` wurde `true`, `{"a":1} {"b":2}` wurde `{"a":1}`. Die dokumentierte Regel (kein JSON → als String-Literal) griff dadurch nicht.
- **Fix:** Ein `ObjectReader` mit `FAIL_ON_TRAILING_TOKENS`; ein Wert, der nicht genau ein JSON-Wert ist, wird wie dokumentiert zum String-Literal.
- **Tests:** `ParameterJsonSerializerTest.testBuildRawValueObject_trailingTokensFallBackToString`, `InferenceParametersTest.aChatTemplateKwargHoldingTwoValuesIsKeptAsText` (beide öffentlichen Setter).
- **Rot-Grün-Nachweis:** Ohne den Fix rot (`{"two":true,…,"object":{"a":1}}`); mit ihm grün (40 + 114).
- **Status:** gefixt
- **Commit:** java-llama.cpp `266d70d`

<a id="f204"></a>
### F204 – java-llama.cpp: `throw_utf8_impl`/`new_string_utf8_impl` rufen JNI über einer schon anstehenden Java-Exception auf und ersetzen sie
- **Zeit:** 2026-10-20 12:50 UTC
- **Datei:** `llama/src/main/cpp/jni_utf8.hpp`, `llama/src/test/cpp/test_jni_helpers.cpp`
- **Schwere:** niedrig bis mittel (die genauere Exception geht verloren; unter Android-CheckJNI bricht ein `FindClass`/`NewObject` bei anstehender Exception den Prozess ab)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20; `jni_utf8.hpp` entstand mit dem Modified-UTF-8-Fix, `jni_guard_impl` hatte die Regel schon.
- **Beschreibung:** Nach einem fehlgeschlagenen JNI-Aufruf (z. B. `GetStringUTFChars` mit `OutOfMemoryError`) rief ein Aufrufer `throw_utf8_impl`: der ASCII-Pfad `ThrowNew` ersetzte die anstehende Exception, der Nicht-ASCII-Pfad rief `FindClass`, `GetMethodID` und `NewObject` — laut JNI-Spezifikation bei anstehender Exception verboten. `jni_guard_impl` lässt eine anstehende Exception stehen, diese beiden Helfer nicht.
- **Fix:** Beide prüfen zuerst `ExceptionCheck()`: `throw_utf8_impl` kehrt zurück und lässt die anstehende Exception stehen, `new_string_utf8_impl` gibt `nullptr` zurück (sein Vertrag: nullptr mit anstehender Exception).
- **Tests:** C++ `JniUtf8.APendingExceptionIsLeftInPlaceOnBothPaths` (ASCII- und 4-Byte-Pfad, kein `ThrowNew`, kein String gebaut, nichts geworfen; `new_string_utf8_impl` ohne `NewStringUTF`).
- **Rot-Grün-Nachweis:** Ohne den Fix rot; mit ihm grün, ctest 645/645.
- **Status:** gefixt
- **Commit:** java-llama.cpp `2611a38`

<a id="f205"></a>
### F205 – java-llama.cpp: Ollama `/api/generate` ohne Prompt (das Vorladen eines Modells) erzeugt eine Antwort auf einen leeren Nutzer-Turn
- **Zeit:** 2026-10-20 13:05 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OllamaApiSupport.java` (neu `isLoadOnly`, `toOllamaLoadResponse`), `OpenAiCompatServer.java` (`handleOllamaGenerate`)
- **Schwere:** niedrig (unnötige Generierung bis `num_predict`/Kontextende bei jedem Vorladen; Clients erwarten `done_reason "load"`)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20; vorher erzeugte der Pfad ohne `raw` (Chat-Template seit diesem Branch) bzw. der `raw`-Pfad eine Antwort auf einen leeren Prompt.
- **Beschreibung:** Ollama beantwortet ein `/api/generate` ohne Prompt mit dem Laden des Modells (`done: true`, leere `response`, `done_reason "load"`) — so laden Clients ein Modell vor. Hier lief ein leerer Nutzer-Turn durch das Chat-Template und wurde beantwortet.
- **Fix:** Ohne Prompt und ohne `suffix` (ein leerer Prompt mit `suffix` bleibt FIM am Dateianfang) antwortet die Route sofort mit Ollamas Ladeantwort, ohne das Backend zu rufen — auch bei `stream` ein einzelnes Objekt, wie Ollama. Entladen (`keep_alive: 0`) gibt es hier nicht; das Modell bleibt geladen.
- **Tests:** `OpenAiCompatServerHttpTest.ollamaGenerateWithoutAPromptOnlyReportsTheModelLoaded` (drei Formen, das Backend wird nicht gerufen), `OllamaApiSupportTest.aGenerateRequestWithoutAPromptOrSuffixOnlyLoadsTheModel`.
- **Rot-Grün-Nachweis:** Ohne den Fix der HTTP-Test rot (das Backend generierte); mit ihm grün (64 + 37).
- **Status:** gefixt
- **Commit:** java-llama.cpp `6c176d5`

## Offene Verdachte ohne Fund-ID

- *(geschlossen 2026-10-19, ohne Fund-ID, Entscheidung des Nutzers: bewusst nicht ändern)* Continue entscheidet die FIM-Fähigkeit eines Ollama-Modells an `{{ .Suffix }}` im `template` von `/api/show`; wir liefern `""`. Ein Platzhalter-Template wäre erfunden (der Server kennt das Prompt-Format des Modells nicht) und für die meisten Modelle falsch. Continue nutzt dann eigene Vorlagen; für die FIM-Token des Modells gibt es `POST /infill` (Continues `llama.cpp`-Provider). README-Hinweis ergänzt.
