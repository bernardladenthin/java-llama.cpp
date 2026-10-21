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
- [F18](#f18) — mittel — *gefixt (Fix `5df523f`; roter Regressionstest nachgereicht, bis dahin „nur beschrieben“, siehe `d5c2622`)*
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
- [F213](#f213) — niedrig — *gefixt*
- [F214](#f214) — niedrig — *gefixt*
- [F215](#f215) — niedrig — *gefixt*
- [F216](#f216) — niedrig — *gefixt*
- [F217](#f217) — niedrig — *gefixt*
- [F218](#f218) — mittel — *gefixt*
- [F219](#f219) — niedrig — *gefixt*
- [F220](#f220) — niedrig — *gefixt*
- [F221](#f221) — mittel — *gefixt*
- [F222](#f222) — niedrig — *gefixt*
- [F223](#f223) — niedrig — *gefixt*
- [F224](#f224) — niedrig — *gefixt*
- [F225](#f225) — niedrig — *gefixt*
- [F226](#f226) — niedrig — *gefixt*
- [F227](#f227) — niedrig — *gefixt*
- [F228](#f228) — niedrig — *gefixt*
- [F229](#f229) — niedrig — *gefixt*
- [F230](#f230) — niedrig — *gefixt*
- [F231](#f231) — niedrig — *gefixt*
- [F232](#f232) — niedrig — *gefixt*
- [F233](#f233) — niedrig — *gefixt*
- [F234](#f234) — niedrig — *gefixt*
- [F235](#f235) — niedrig — *gefixt*
- [F236](#f236) — niedrig — *gefixt*
- [F237](#f237) — niedrig — *gefixt*
- [F238](#f238) — niedrig — *gefixt*
- [F239](#f239) — niedrig — *gefixt*
- [F240](#f240) — niedrig — *gefixt*
- [F241](#f241) — niedrig bis mittel — *gefixt*
- [F242](#f242) — niedrig — *gefixt*
- [F243](#f243) — niedrig — *gefixt*

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
- **Commit:** java-llama.cpp `aaa7b5a`
- **Doku-Lücke aus diesem Fix (2026-10-17, nur beschrieben):** Seit dem Fix wirft `TrainingParameters.toJson()` – und damit beide `LlamaTrainer.finetune`-Überladungen – bei ungültiger Konfiguration `IllegalArgumentException`; die Javadoc beider `finetune` nennt nur `LlamaException`. Reine Doku, kein Test kann sie rot machen, daher nach der Nutzerregel unverändert.
- **Nachtrag (Branch-Review 2026-10-20, Fund J8-5):** Die Doku-Lücke ist geschlossen. `TrainingParameters#toJson` nennt `@throws IllegalArgumentException` (weniger als eine Epoche, `valSplit` außerhalb `[0, 1)`, nicht genau ein Korpus) und `@throws NullPointerException` (`modelPath`/`outputPath` am Builder nicht gesetzt; die Meldung nennt das Feld -- die NPE gab es schon vor dem Fix, er gab ihr nur den Namen). `LlamaTrainer#finetune(TrainingParameters)` nennt beide mit Verweis auf `toJson`; die Bequemlichkeits-Überladung nur `IllegalArgumentException` für `epochs` unter 1, die einzige dort erreichbare Prüfung (Text als Parameter, keine Datei, `valSplit` auf dem Standard, Pfade nicht null unter `@NullMarked`). Nur Doku; die Prüfungen pinnt `TrainingParametersTest`.

<a id="f05"></a>
### F05 – java-llama.cpp: `ChatRequest.buildToolsJson()` kürzt ein Tool-Schema still auf den ersten JSON-Wert
- **Zeit:** 2026-10-16 23:20 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/parameters/ChatRequest.java` (`buildToolsJson`, Zeile mit `SCHEMA_READER.readTree`)
- **Schwere:** niedrig
- **Beschreibung:** Das Schema jeder `ToolDefinition` wurde mit einfachem `readTree` gelesen; das liest den ersten JSON-Wert und ignoriert den Rest. `{"type":"object"} {"required":["x"]}` wurde so ohne Fehler als `{"type":"object"}` gesendet – ein anderes Schema als vom Aufrufer geschrieben. `JsonParameters` nutzt aus genau diesem Grund `FAIL_ON_TRAILING_TOKENS` (siehe CLAUDE.md), `ChatRequest` umging das. Jetzt gleicher strikter Reader → solche Strings nehmen den vorhandenen Fallback für ungültiges JSON.
- **Weitere Beobachtung (nicht geändert):** Ungültiges JSON wird absichtlich als **String** unter `parameters` weitergereicht, `""` und Leerzeichen werden zu `"parameters":null`. Ob eine `IllegalArgumentException` (wie in `JsonParameters`) besser wäre, ist eine API-Entscheidung.
- **Reproduktion:** `ChatRequestTest$…#buildToolsJsonDoesNotSilentlyTruncateASchemaWithTrailingContent` (vorher rot, danach grün), plus Positivtest `buildToolsJsonEmbedsAValidSchemaAsAnObject`; `ChatRequestTest` 24/24 grün. `spotbugs:check` nicht gelaufen.
- **Nachtrag (Branch-Review 2026-10-20):** Die „API-Entscheidung“ ist mit F233 gefallen: `appendTool` weist ein Schema, das nicht genau ein JSON-Wert ist, mit `IllegalArgumentException` samt Werkzeugnamen ab; der String-Rückfall ist weg, der Test oben durch `appendToolRejectsASchemaThatIsNotExactlyOneJsonValue` ersetzt.
- **Status:** gefixt
- **Commit:** java-llama.cpp `d6a9be5`

<a id="f06"></a>
### F06 – java-llama.cpp: `RouterClient.awaitModelLoaded(id, Long.MAX_VALUE)` bricht sofort mit Timeout ab
- **Zeit:** 2026-10-16 23:35 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/RouterClient.java` (`awaitModelLoaded`, Deadline-Berechnung)
- **Schwere:** mittel
- **Beschreibung:** `System.currentTimeMillis() + timeoutMillis` läuft bei `Long.MAX_VALUE` („so lange wie nötig“, eine naheliegende Aufrufform) über und wird negativ. Die Schleife pollt nie und wirft „did not reach LOADED within 9223372036854775807 ms (last status: null)“, auch wenn das Modell längst geladen ist. Fix: Deadline sättigt bei `Long.MAX_VALUE`.
- **Reproduktion:** `RouterClientTest#awaitModelLoaded_longMaxValueTimeoutStillPolls` (vorher rot mit genau dieser Meldung, danach grün)
- **Status:** gefixt
- **Commit:** java-llama.cpp `a4e5b22`

<a id="f07"></a>
### F07 – java-llama.cpp: `RouterClient` mit IPv6-Literal (`"::1"`) kann nie verbinden
- **Zeit:** 2026-10-16 23:40 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/RouterClient.java` (`request`, URL-Aufbau)
- **Schwere:** niedrig
- **Beschreibung:** Die URL wurde als `"http://" + host + ":" + port` gebaut. Für `new RouterClient("::1", port)` (Router mit `--host ::1`; Javadoc: „host name or address“) entsteht `http://::1:port/models` → `MalformedURLException: Error at index 0 in ":1:8080"` vor jedem Verbindungsversuch. Workaround war `"[::1]"`. Fix: IPv6-Literal wird geklammert, bereits geklammerte Hosts bleiben unverändert. Nicht abgedeckt: Zonen-IDs (`fe80::1%eth0`) bräuchten zusätzlich `%25`-Kodierung. `toString()` zeigt den Host weiterhin ungeklammert.
- **Reproduktion:** `RouterClientTest#ipv6LiteralHostIsBracketedInTheRequestUrl` (vorher rot: `MalformedURLException`, danach andere `IOException`, weil auf Port 1 nichts lauscht). Der Sandbox fehlt IPv6, ein echter Server auf `::1` war hier nicht testbar.
- **Status:** gefixt
- **Commit:** java-llama.cpp `5b98dc7`
- **Nachtrag (Branch-Review 2026-10-20, Fund J7-3):** `toString()` zeigt den Host jetzt wie die Anfrage-URL (`urlHost`): `RouterClient("::1", 8080)` wird zu `RouterClient(http://[::1]:8080)` statt der ungültigen Autorität `http://::1:8080`; die Klassen-Javadoc („renders as its target URL“) stimmt damit. Test `RouterClientTest#toStringBracketsAnIpv6HostAsTheRequestUrlDoes` (auch `"[::1]"` bleibt einfach geklammert); ohne die Änderung rot („Expected: is "RouterClient(http://[::1]:8080)" but: was "RouterClient(http://::1:8080)"“), mit ihr grün (`RouterClientTest` 22/22).
- **Nachtrag (Branch-Review 2026-10-20, Fund J7-1):** „bereits geklammerte Hosts bleiben unverändert“ hat jetzt einen Test über den Anfrageweg: `RouterClientTest#alreadyBracketedIpv6HostIsNotBracketedAgain` (`new RouterClient("[::1]", 1)`, `listModels()` wirft eine `IOException`, keine `MalformedURLException`). Ohne `!host.startsWith("[")` entsteht `http://[[::1]]:1/models`, das `URI.create` als `IllegalArgumentException` ablehnt – außerhalb des `IOException`-Vertrags von `request()`. Gegenprobe rot („Unexpected exception type thrown, expected: <java.io.IOException> but was: <java.lang.IllegalArgumentException>“); am HEAD fängt dieselbe Mutation auch der `toString()`-Test aus J7-3, weil beide `urlHost` teilen; der neue Test pinnt den Vertrag der Anfrage selbst. `RouterClientTest` 23/23 grün.

<a id="f09"></a>
### F09 – java-llama.cpp: Loader löscht fremde `llama*`/`ggml*`-Einträge im Temp-Verzeichnis, Verzeichnisse rekursiv
- **Zeit:** 2026-10-16 23:30 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/loader/LlamaLoader.java` (`isStaleExtraction` / `cleanup` / `cleanPath`)
- **Schwere:** hoch (Datenverlust außerhalb der eigenen Dateien)
- **Beschreibung:** `cleanup()` lief bei jedem Start, der die Natives-Jars erreicht, über `java.io.tmpdir` (bzw. `net.ladenthin.llama.tmpdir`) und löschte **sofort und rekursiv** jeden Eintrag, dessen Name mit `jllama`, `llama` oder `ggml` beginnt und kein `jllama-backend-*`-Verzeichnis ist. Gedacht war das für die einzelnen Bibliotheksdateien des flachen Layouts vor 6.0.0 (so steht es in der Javadoc). Getroffen wurde aber auch z. B. ein `/tmp/llama.cpp`-Checkout, ein `/tmp/ggml-models/`-Ordner oder ein heruntergeladenes `/tmp/llama-2-7b.Q4_K_M.gguf`. Fix: Aus dem alten Layout werden nur noch **reguläre Dateien mit Bibliotheksendung** gelöscht (`.so`, `.dll`, `.dylib`, `.jnilib`, `.metal`).
- **Reproduktion:** `LlamaLoaderTest#staleExtractionLeavesForeignEntriesWithAMatchingPrefixAlone` (vorher rot, danach grün); `LlamaLoaderTest` 63/63, inkl. der bestehenden Fälle `jllama-3f2a.so` / `ggml-metal.metal` (werden weiter gelöscht).
- **Bitte prüfen:** Die genauen Dateinamen des alten Layouts konnte ich nicht verifizieren, weil der Clone flach ist und keine Historie vor 6.0.0 enthält. Falls es damals auch Endungen wie `.tmp` oder `.lck` gab, bleiben die jetzt liegen. Das ist harmlos, nur Restmüll.
- **Nachtrag (Branch-Review 2026-10-20):** Am Tag `v5.1.0` geprüft: das alte Layout entpackte `System.mapLibraryName("jllama")` flach (`libjllama.so`, `libjllama.dylib`, `jllama.dll`), unter macOS dazu `ggml-metal.metal`, jede Kopie über `Files.createTempFile(dir, name + ".", ".tmp")`. `libjllama.so`/`libjllama.dylib` trugen nie ein Bereinigungspräfix und wurden von keiner Version entfernt; was dieser Fix entfernt, sind `jllama.dll` (und ältere `llama*`/`ggml*`-DLLs) und `ggml-metal.metal`. Die `<name>.<Ziffern>.tmp`-Reste einer abgebrochenen Kopie blieben seit diesem Fix liegen; seit F239 gehen sie wieder, sobald sie `STALE_EXTRACTION_AGE` alt sind. Das „Bitte prüfen“ ist damit erledigt.
- **Nachtrag (Branch-Review 2026-10-20):** Die Endungsliste und der Reguläre-Datei-Wächter waren nur zum Teil getestet: `.dll`, `.dylib` und `.jnilib` ließen sich aus `FLAT_LAYOUT_SUFFIXES` streichen und `NOFOLLOW_LINKS` weglassen, ohne dass etwas rot wurde (Fund J10-3). Neu in `LlamaLoaderTest`: `#flatLayoutLibraryNamesAreTheLibrarySuffixes` (alle fünf Endungen ja, `.gguf`/`.log`/`.tmp`/`llama.cpp` nein), `#staleExtractionKeepsADirectoryNamedLikeALibrary` (ein Verzeichnis `jllama.dll` mit Inhalt bleibt) und `#staleExtractionKeepsASymlinkNamedLikeALibrary` (ein Link `ggml.dll` auf eine fremde Datei bleibt; übersprungen, wo keine Links angelegt werden dürfen). Gegenprobe rot: ohne den Wächter die beiden neuen Fälle und der Verzeichnisfall des F239-Tests (`expected: <false> but was: <true>`; seit F239 steht der Wächter vor beiden Zweigen, deckte also den Zweig ohne `.tmp` schon mit), mit einem link-folgenden `isRegularFile(path)` der Link-Test, ohne `.dll`/`.dylib`/`.jnilib` je der Endungstest (`jllama.dll ==> expected: <true> but was: <false>` usw.); unverändert 67/67 grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `61ea8f5`

<a id="f12"></a>
### F12 – java-llama.cpp: Ollama-`/api/chat` verwirft Bilder (`images`), obwohl `/api/show` „vision“ meldet
- **Zeit:** 2026-10-16 23:55 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OllamaApiSupport.java` (`toOpenAiMessage`)
- **Schwere:** mittel (Feature stillschweigend wirkungslos)
- **Beschreibung:** Ollama-Clients schicken Bilder als `"images": ["<base64>", …]` neben `"content"`. `toOpenAiMessage` kopierte die Nachricht unverändert; der OpenAI-Parser von llama.cpp liest nur `image_url`-Content-Parts, das Feld `images` wurde also ignoriert. Das Vision-Modell antwortete ohne Bild, kein Fehler. Gleichzeitig meldet `/api/show` die Fähigkeit `vision`, wenn `--mmproj` gesetzt ist. Fix: `images` wird entfernt und als `image_url`-Parts (Data-URL, MIME-Typ aus den Magic-Bytes: JPEG/GIF/WebP, sonst PNG – llama.cpp dekodiert nach Inhalt) hinter dem Text angehängt.
- **Reproduktion:** `OllamaApiSupportTest#toOpenAiChatRequestTurnsOllamaImagesIntoImageUrlParts` (vorher rot, danach grün); `OllamaApiSupportTest` 20/20, `OpenAiCompatServerHttpTest` 43/43. Ende-zu-Ende mit echtem Vision-Modell nicht geprüft (kein Modell im Sandbox).
- **Nebenbefund (nicht geändert):** Die gestreamte Abschlusszeile (`toOllamaDoneLine`) setzt `done_reason` immer auf `"stop"`, auch bei Abbruch durch `length`; die nicht-gestreamte Antwort übernimmt `finish_reason` korrekt.
- **Nachtrag (Branch-Review 2026-10-20):** Die MIME-Erkennung aus den Magic-Bytes und der Durchlass einer fertigen `data:`-URL hatten keinen Test; der PNG-Test oben nimmt nur den `else`-Zweig (Fund J3-6). `OllamaApiSupportTest#ollamaImageMimeTypeFollowsTheMagicBytes` pinnt `/9j/` → `image/jpeg`, `R0lGOD` → `image/gif`, `UklGR` → `image/webp` und den PNG-Rückfall; `#ollamaImageThatIsAlreadyADataUrlIsPassedThrough` den Durchlass — der einzige Zweig mit sichtbarer Wirkung, denn ein doppeltes `data:` lehnt llama.cpp ab, während es den Untertyp nie liest. Gegenprobe rot: vertauschte JPEG/GIF-Literale (`Expected: is "data:image/jpeg;base64,/9j/4AAQ" but: was "data:image/gif;base64,/9j/4AAQ"`), fehlender WebP-Zweig (`… but: was "data:image/png;base64,UklGRh4A"`), fehlender Durchlass (`… but: was "data:image/png;base64,data:image/jpeg;base64,/9j/4AAQ"`); unverändert 46/46 in der Klasse grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `ef7295c`

<a id="f13"></a>
### F13 – java-llama.cpp: `setRopeScaling(UNSPECIFIED | LONGROPE | MAX_VALUE)` macht das Modell unladbar
- **Zeit:** 2026-10-17 00:05 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/parameters/ModelParameters.java` (`setRopeScaling`), `args/RopeScalingType.java`
- **Schwere:** mittel
- **Beschreibung:** llama.cpp akzeptiert bei `--rope-scaling` nur `none`, `linear`, `yarn` (`common/arg.cpp` @ b11639, Zeile 2337ff.) und wirft sonst `invalid value` → `LlamaModel` lädt nicht („Failed to parse model parameters“). `RopeScalingType` bietet zusätzlich `UNSPECIFIED` (laut Javadoc „use the model default“), `LONGROPE` und `MAX_VALUE`; der Setter reichte sie ungeprüft als `unspecified`/`longrope`/`maxvalue` durch. Genau die Fehlerklasse aus CLAUDE.md („a Java test asserting the string mapping passes forever“): `testSetRopeScalingAllValues` prüfte nur die String-Abbildung aller Konstanten. Gegencheck aller anderen `args`-Enums gegen den Upstream-Parser (CacheType, DraftSampling, FlashAttn, GpuSplitMode, LazyMode, LoadMode, MiroStat, NumaStrategy, PoolingType, ReasoningFormat, Sampler): alle Werte akzeptiert.
- **Fix:** `UNSPECIFIED` entfernt die Option (wie `setPoolingType`), `LONGROPE`/`MAX_VALUE` → `IllegalArgumentException` beim Aufruf. Kein vorher funktionierender Pfad geht verloren (diese drei Werte scheiterten ohnehin beim Laden). Die Enum-Konstanten selbst bleiben (API-Kompatibilität) – ob sie gelöscht werden sollen, ist deine Entscheidung (CLAUDE.md: „A name with no counterpart is deleted, never deprecated“).
- **Reproduktion:** `ModelParametersExtendedTest#testSetRopeScalingUnspecifiedLeavesTheModelDefault`, `#testSetRopeScalingRejectsValuesTheParserDoesNotAccept` (beide vorher rot), `#testSetRopeScalingAllValuesTheParserAccepts` (ersetzt den alten Test). `WireNameRegistryTest`-Treiber nimmt bei Ablehnung die nächste Konstante. Grün: ModelParametersExtendedTest 148, ModelParametersTest 92, WireNameRegistryTest 8, RopeScalingTypeTest 24. Nicht gelaufen: C++-`test_model_flags` (prüft nur Optionsnamen, nicht Werte), `spotbugs:check`.
- **Status:** gefixt
- **Commit:** java-llama.cpp `abe776c`

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
- **Commit:** java-llama.cpp `6ed949e`

<a id="f17"></a>
### F17 – java-llama.cpp: Responses-API (`POST /v1/responses`) verwirft `input_image`-Teile → Vision-Modell sieht kein Bild
- **Zeit:** 00:10 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/ResponsesApiSupport.java:138` (`appendInput`, Fall `message`)
- **Schwere:** mittel
- **Beschreibung:** Der Translator flachte jedes Content-Array einer `message` auf seinen Text ab (`inputContentText` liest nur `text`). Ein `{"type":"input_image","image_url":"data:…"}`-Teil kam nie im Chat-Kern an; ein Vision-Modell antwortete, als wäre kein Bild geschickt worden – still, ohne Fehler. Gleiche Fehlerklasse wie der Ollama-`images`-Fund oben.
- **Fix:** Enthält das Array `input_image`-Teile, wird ein OpenAI-Content-Part-Array gebaut (`text`-Teile + `image_url`-Teile; `image_url` als String oder `{url}`). Reine Text-Nachrichten bleiben unverändert ein String. `file_id`-Bilder (ohne URL) werden weiterhin ignoriert.
- **Reproduktion:** `ResponsesApiSupportTest#requestTurnsInputImagePartsIntoImageUrlParts` (vorher rot, nachher 10/10 grün)
- **Nachtrag (Branch-Review 2026-10-20):** Der Satz „`file_id`-Bilder (ohne URL) werden weiterhin ignoriert“ gilt nicht mehr: ein `input_image` ohne URL wird seit F224 wie upstream mit 400 abgewiesen.
- **Status:** gefixt
- **Commit:** java-llama.cpp `558422a`

<a id="f18"></a>
### F18 – java-llama.cpp: `NativeServer.close()` nicht atomar → doppeltes Freigeben des nativen Servers bei parallelen Aufrufen
- **Zeit:** 00:17 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/NativeServer.java:284` (`close`, ebenso `isRunning`, `start`)
- **Schwere:** mittel
- **Beschreibung:** `close()` las `handle`, prüfte auf 0 und setzte es danach auf 0 – ohne Synchronisation. Zwei gleichzeitige `close()` (Shutdown-Hook + try-with-resources, oder in `NativeServer.main` der Hook und der `finally`-Block, wenn der Worker genau beim Ctrl-C selbst endet) kamen beide durch und riefen `stopNativeServer(h)` doppelt: zweimal `join()` und zweimal `delete` desselben `native_server` → JVM-Absturz. Ebenso konnte `isRunning()` einen Handle abfragen, den ein paralleles `close()` gerade freigab (Use-after-free). Der Kommentar in `main` („the finally and the hook both firing is safe“) gilt nur nacheinander.
- **Fix:** privates `lifecycleLock` serialisiert `start`, `isRunning` und `close` (keine `synchronized`-Methoden, SpotBugs USO_UNSAFE_METHOD_SYNCHRONIZATION). `spotbugs:check` grün, `NativeServerSmokeTest` + `LlamaArchitectureTest` grün.
- **Reproduktion:** `NativeServerLifecycleTest` über die neue paketinterne Naht `NativeServer.Backend` (ein Fake statt der JNI-Aufrufe, damit kein erfundener Handle libjllama erreicht): `close_whileIsRunningIsInsideTheNativeCall_stopsOnlyAfterItReturned`, `close_whileStartIsStillStarting_stopsTheServerOnceItStarted`, `close_calledTwice_stopsTheHandleOnce`. Der Fake hält `isRunning` bzw. `start` im „nativen“ Aufruf fest, dadurch sind beide Race-Fenster deterministisch. (Die alte Probe `/home/user/BUGHUNT-2026-10-17-F18-probe.java`: vorher 4 / 20 000 doppelte Stopps, nachher 0 / 20 000.)
- **Rot-Grün-Nachweis:** Gegen den Lebenszyklus vor `5df523f` (ohne `lifecycleLock`, Naht beibehalten) **rot**, 2 von 3: „the handle was stopped while isRunning() still used it – Expected: is <false> but: was <true>“ und „close() returned while the server was still starting, so it kept running – Expected: is <1> but: was <0>“. Mit Fix **grün** 3/3 (skipped 0), dreimal hintereinander, ebenso `NativeServerSmokeTest` 10/10, `NativeServerWorkerCommandTest` 5/5, `LlamaArchitectureTest` 13/13, `spotbugs:check` grün. Das doppelte `close()` aus zwei Threads selbst ist mit der Naht nicht steuerbar (das Fenster liegt zwischen Lesen und Nullsetzen des Feldes); `close_calledTwice_stopsTheHandleOnce` deckt nur den sequentiellen Fall ab.
- **Status:** gefixt (Fix `5df523f`; roter Regressionstest nachgereicht, bis dahin „nur beschrieben“, siehe `d5c2622`)
- **Commit:** java-llama.cpp `5df523f` (Fix), Test und Naht `8845eca`
- **Nachtrag (Branch-Review 2026-10-20, Fund J7-2):** Die drei Wächter von `start()` (älter als dieser Fix, erst über die Naht ohne Modell testbar) haben jetzt Tests in `NativeServerLifecycleTest`: `#start_whenTheBackendThrowsALlamaException_freesTheSingleInstanceSlot` und `#start_whenTheBackendThrowsAnError_freesTheSingleInstanceSlot` (beide Arme von `catch (RuntimeException | Error)`, `UnsatisfiedLinkError` wie aus `LlamaLoader.initialize()`), `#start_calledTwiceOnTheSameInstance_throwsIllegalStateException` („already started“, der Fake einmal gestartet, danach genau ein `stop`) und `#start_whileAnotherServerRuns_throwsIllegalStateException` („another NativeServer“, der zweite Fake nie gestartet, ein dritter ebenso abgewiesen, der Platz also weiter belegt). `FakeBackend` bekam dafür `starts` und `startFailure`. Gegenprobe rot: ohne `RUNNING.set(false)` im `catch` („Unexpected exception type thrown, expected: <java.lang.UnsatisfiedLinkError> but was: <java.lang.IllegalStateException>“, die folgenden Tests der Klasse kaskadieren, weil der Platz prozessweit belegt bleibt), ohne die Handle-Prüfung (`containsString("already started")` scheitert an der Meldung des anderen Wächters), ohne den CAS („Expected java.lang.IllegalStateException to be thrown, but nothing was thrown.“); unverändert 7/7 grün.
- **Nachtrag (Branch-Review 2026-10-20, Fund J7-4):** Der Kommentar in `main` nannte noch die Null-Prüfung des Handles als Grund, warum `finally` und Shutdown-Hook beide `close()` rufen dürfen. Er sagt jetzt, dass `close()` idempotent **und** über `lifecycleLock` serialisiert ist, beide also gleichzeitig feuern dürfen (Shutdown-Hooks laufen in eigenem Thread) und der native Server genau einmal gestoppt wird. Reiner Kommentar, kein Test.

<a id="f24"></a>
### F24 – java-llama.cpp: `OpenAiServerCli` bewirbt bei kleinem `-c` fast kein Eingabebudget (`-c 2048` → 1 Token)
- **Zeit:** 05:55 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiServerCli.java:721` (`Options.toServerConfig`)
- **Schwere:** mittel
- **Beschreibung:** Das Ausgabebudget ist `min(2048, ctx/2)`, das Eingabebudget wurde aber als `ctx − 2048` (der Default, nicht das tatsächliche Ausgabebudget) berechnet. Für `ctx < 4096` passt das nicht zusammen: `-c 2048` → Eingabe **1** Token, Ausgabe 1024; `-c 1024` → beworbenes Fenster 513 statt 1024; `-c 3000` → 2452. Das Eingabebudget landet in `maxInputTokens` (Copilot-Konfigurationsausgabe) und in der beworbenen Kontextlänge (`/api/show`, `/props`: `maxInputTokens + maxOutputTokens`); Clients, die ihren Prompt danach zuschneiden, schicken praktisch nichts. Für `-c ≥ 4096` (README-Beispiel 8192 → 6144) war das Ergebnis zufällig richtig.
- **Fix:** `maxInputTokens = max(1, ctx − maxOutputTokens)` mit dem tatsächlich gesetzten Ausgabebudget; 8192 → 2048/6144 unverändert.
- **Reproduktion:** `OpenAiServerCliTest#smallContextSizeSplitsTheWholeWindowBetweenInputAndOutput` (ctx 1024/2048/3000; vorher `Expected 1024 but was 513`, nachher 38/38 grün).
- **Status:** gefixt
- **Commit:** java-llama.cpp `2cf3640`

<a id="f25"></a>
### F25 – java-llama.cpp: `OpenAiCompatServer` mit `--api-key` lehnt jeden Anthropic-Client ab (`x-api-key`) und `bearer` in Kleinschreibung
- **Zeit:** 06:03 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiCompatServer.java:893` (`authorized`; gefunden beim Review von `OpenAiServerConfig`)
- **Schwere:** mittel
- **Beschreibung:** Akzeptiert wurde nur `Authorization: Bearer <key>` mit exakt großgeschriebenem `Bearer`. Die Anthropic-Messages-Oberfläche (`POST /v1/messages`) wird von Anthropic-SDKs aber per `x-api-key`-Header authentifiziert – mit gesetztem `--api-key` bekam jeder solche Client 401, die Anthropic-Oberfläche war damit nur ohne Schlüssel nutzbar. Zudem ist der Schema-Name laut RFC 7235 case-insensitiv; `bearer <key>` wurde abgelehnt.
- **Fix:** `x-api-key` wird zusätzlich akzeptiert, das Schema case-insensitiv geprüft; beide Vergleiche weiterhin zeitkonstant (`MessageDigest.isEqual`). `spotbugs:check` grün.
- **Reproduktion:** `OpenAiCompatServerHttpTest#anthropicApiKeyHeaderAndLowercaseBearerSchemeAreAccepted` (vorher 401 statt 200, nachher Klasse 44/44 grün).
- **Status:** gefixt
- **Commit:** java-llama.cpp `78fd0af`

<a id="f26"></a>
### F26 – java-llama.cpp: `withTokenIdBias`/`withTokenBias` mit `Float.NEGATIVE_INFINITY` sperrt das Token nicht – wird still ignoriert
- **Zeit:** 06:09 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/parameters/ParameterJsonSerializer.java:236` (`buildTokenIdBiasArray`/`buildTokenStringBiasArray`; gefunden beim Review von `InferenceParameters`)
- **Schwere:** mittel
- **Beschreibung:** JSON kennt kein Unendlich; Jackson schreibt einen nicht-endlichen `float` als String: `{"logit_bias":[[5,"-Infinity"]]}` (gemessen). llama.cpps Parser (`server-schema.cpp` b11639, `parse_bias`) akzeptiert nur eine Zahl oder `false` und überspringt alles andere per `continue`. Der naheliegende Weg, ein Token zu verbieten (`withTokenIdBias(Map.of(id, Float.NEGATIVE_INFINITY))`), bewirkte damit gar nichts – ohne Fehler. `NaN` und `+Infinity` ebenso.
- **Fix:** `-Infinity` wird als `false` (llama.cpps Schreibweise für ein Verbot, wie schon bei `withDisabledTokenIds`) gesendet, `NaN`/`+Infinity` werfen `IllegalArgumentException`.
- **Reproduktion:** `ParameterJsonSerializerTest#testBuildTokenIdBiasArray_negativeInfinityBansTheToken`, `#testBuildTokenIdBiasArray_nanOrPositiveInfinityIsRejected` (vorher rot, nachher 34/34); `InferenceParametersTest` 105/105, `WireNameRegistryTest` 8/8 grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `d38840e`
- **Nachtrag (Branch-Review 2026-10-20, Fund J8-6):** Die öffentliche Javadoc nannte das Verhalten nicht, nur die private `addBias`. `ParameterJsonSerializer#buildTokenIdBiasArray`/`#buildTokenStringBiasArray` und `InferenceParameters#withTokenIdBias`/`#withTokenBias` sagen jetzt, dass `Float.NEGATIVE_INFINITY` als llamas Sperrform `false` geht, und nennen `@throws IllegalArgumentException` für NaN und +Unendlich. Nur Doku; `ParameterJsonSerializerTest` pinnt beides.

<a id="f28"></a>
### F28 – java-llama.cpp: `InferenceParameters.withMessages(List<ChatMessage>)` verwirft `tool_calls` und `tool_call_id`
- **Zeit:** 06:22 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/parameters/ParameterJsonSerializer.java:152` (`buildMessages(List<ChatMessage>)`)
- **Schwere:** mittel
- **Beschreibung:** Der gemeinsame Serializer schrieb nur `role` und `content` (bzw. Content-Parts). `ChatRequest.buildMessagesJson` ergänzte `tool_calls`/`tool_call_id` nachträglich selbst – `InferenceParameters.withMessages(List<ChatMessage>)` nutzt den Serializer aber direkt. Eine Tool-Calling-Runde über diesen Weg (`ChatMessage.assistantToolCalls(...)`, `ChatMessage.toolResult(...)`) schickte einen Assistant-Turn ohne seine Aufrufe und ein Tool-Ergebnis ohne Zuordnung: `[{"role":"assistant","content":""},{"role":"tool","content":"42"}]` (gemessen).
- **Fix:** die beiden Felder schreibt jetzt der Serializer; die Duplikat-Schleife in `ChatRequest` entfällt (Ausgabe dort identisch). `spotbugs:check` grün.
- **Reproduktion:** `InferenceParametersTest#withChatMessagesKeepsToolCallsAndToolCallIds` (vorher rot, nachher 106/106); `ChatRequestTest` 24/24, `ParameterJsonSerializerTest` 34/34, `OpenAiCompatServerHttpTest` 44/44 grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `2c32c84`

<a id="f29"></a>
### F29 – java-llama.cpp: Anthropic-Messages-API (`POST /v1/messages`) verwirft `image`-Blöcke → Vision-Modell sieht kein Bild
- **Zeit:** 06:29 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/AnthropicApiSupport.java:151` (`appendOpenAiMessages`)
- **Schwere:** mittel
- **Beschreibung:** Gleiche Fehlerklasse wie F17 (Responses) und der Ollama-`images`-Fund: der Translator kannte nur `text`, `tool_use` und `tool_result`; ein `{"type":"image","source":{"type":"base64","media_type":…,"data":…}}`-Block (so senden Anthropic-Clients Bilder) fiel im `default`-Zweig weg. Ein Vision-Modell hinter `/v1/messages` antwortete, als wäre kein Bild geschickt worden – still.
- **Fix:** Bild-Blöcke werden zu OpenAI-`image_url`-Teilen (Data-URL für `base64`, die URL selbst für `url`-Quellen); eine Nachricht mit Bild bekommt Content als Teile-Array in Blockreihenfolge, Tool-Calls bleiben erhalten. Nachrichten ohne Bild unverändert. `spotbugs:check` grün.
- **Reproduktion:** `AnthropicApiSupportTest#requestTurnsImageBlocksIntoImageUrlParts` (vorher rot, nachher 12/12); `OpenAiCompatServerHttpTest` 44/44 grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `3d38dd0`
- **Nachtrag (Branch-Review 2026-10-20, Fund J4-5):** Die `url`-Quelle und der Rückfall für eine andere Quelle hatten keinen Test (nur `base64`). `AnthropicApiSupportTest#requestTurnsUrlImageSourcesIntoImageUrlParts` pinnt die URL selbst als `image_url`, in einer Nachricht und in einem `tool_result`; `#anUnsupportedImageSourceLeavesTheMessageText` (Quelle `ftp`) die Textform der Nachricht und den Text eines `tool_result`, das nur ein solches Bild trägt. Gegenprobe rot: ohne den `url`-Zweig („Expected: is "image_url" but: was """), mit `url`-Übernahme für jede Quelle („Expected: is <true> but: was <false>“); mit dem Code grün (`AnthropicApiSupportTest` 36/36). Die Quelle `file` (Files-API) und die Vorgabe `image/png` statt upstreams `image/jpeg` für ein `base64` ohne `media_type` sind nicht Teil dieses Nachtrags.

<a id="f30"></a>
### F30 – java-llama.cpp: Fehlgeschlagener `RpcServer.start` lässt den Server weiterlaufen, wenn der Stopp vor dem Bind kommt
- **Zeit:** 06:34 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/RpcServer.java:526` (vorher: einmaliges `backend.stop()` in `awaitListening`)
- **Schwere:** mittel
- **Beschreibung:** Läuft der Start in den Timeout (z. B. langsame Geräte-Initialisierung) oder ist der aufrufende Thread unterbrochen, schickt `start` genau **einen** `stop()` und gibt den Single-Instance-Slot frei. Der native Server (patches/0015) setzt aber beim Binden des Sockets `g_rpc_server_stop = false` – ein Stopp vor dem Bind geht verloren. Der Server lauscht danach als verwaister Daemon-Thread weiter, obwohl `start` eine `LlamaException` geworfen hat; ein zweiter `start` belegt dann denselben globalen Server-Zustand. Beim unterbrochenen Aufrufer kehrte `join` zudem sofort zurück, statt zu warten. Fix: `stopAndJoin` wiederholt den Stopp alle 100 ms, solange der Thread lebt (bis `STOP_TIMEOUT_MILLIS`), und stellt das Interrupt-Flag danach wieder her.
- **Reproduktion:** `RpcServerLifecycleTest#aStopThatArrivesBeforeTheBindDoesNotLeaveTheServerRunning` (Fake-Backend, das wie der native Server das Stopp-Flag beim Binden zurücksetzt; ohne Fix rot nach 31 s, mit Fix 15/15 grün)
- **Status:** gefixt
- **Commit:** e824ae5
- **Nachtrag (Branch-Review 2026-10-20):** Die Wiederholung des Stopps war auf `STOP_TIMEOUT_MILLIS` begrenzt; überlebte der Thread sie, gab `start` den Einzelinstanz-Platz trotzdem frei und meldete „could not listen“. Seit F237 behält ein überlebender Thread den Platz, bis er endet, und `start` meldet ihn (`RpcServerLifecycleTest#aServerThatIgnoresTheStopIsReportedAndKeepsTheSlot`).

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
- **Commit:** java-llama.cpp `79fa666`

<a id="f37"></a>
### F37 – java-llama.cpp: Ollama-Tool-Call mit leeren Argumenten kommt als `"arguments": null` statt `{}` an
- **Zeit:** 2026-10-17 17:00 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OllamaApiSupport.java:345` (`parseArgumentsToObject`, genutzt von `toOllamaToolCalls` für `/api/chat` gestreamt und blockierend)
- **Schwere:** niedrig
- **Beschreibung:** Ollama erwartet `tool_calls[].function.arguments` als Objekt. Die Umwandlung aus dem OpenAI-String parste mit `readTree` und fiel nur bei einer `IOException` auf `{}` zurück (so der Kommentar). `readTree("")` wirft aber nicht, sondern liefert einen `MissingNode`, der als `null` serialisiert wird; ebenso ergab ein skalarer Text (`"5"`) einen Skalar. Ein parameterloser Aufruf, für den llama.cpp `"arguments": ""` liefert (oder dessen gestreamte Deltas keine Argumente enthalten, `ToolCallDeltaAccumulator` setzt dann `""`), kam bei Ollama-Clients als `"arguments": null` an. Fix: alles, was nicht zu einem Objekt oder Array parst, wird `{}`.
- **Reproduktion:** `OllamaApiSupportTest#toOllamaChatResponseTurnsEmptyToolCallArgumentsIntoAnEmptyObject` (Jackson-Verhalten vorab per Probe bestätigt: `readTree("")` → `MissingNode` → `"arguments":null`)
- **Rot-Grün-Nachweis:** ohne Fix rot 1/21 („Expected: is <true> but: was <false>“: kein Objekt), mit Fix grün 21/21 (skipped 0); Gegenprobe per `git stash` rot, nach `stash pop` grün. `OpenAiCompatServerHttpTest` 44/44, `LlamaArchitectureTest` 13/13, `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `2940edd`

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
- **Commit:** java-llama.cpp `2a5c835`

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
- **Commit:** java-llama.cpp `6946b77`

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
- **Commit:** java-llama.cpp `4d19a04`

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
- **Commit:** java-llama.cpp `3eb7a6e`

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
- **Commit:** java-llama.cpp `8a13aa9`
- **Nachtrag (Branch-Review 2026-10-20):** Der String-Rückfall war keine Verbesserung: upstream übernimmt `""` als Schema und weist es beim Grammatikbau mit „schema must be an object“ ab, ohne den Werkzeugnamen. Ersetzt durch F233 (Abweisung in `appendTool`); der Test oben ist entfallen.

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
- **Commit:** java-llama.cpp `f3b8aed`
- **Nachtrag (Branch-Review 2026-10-20):** Jede `InvalidPathException` als „fehlt“ zu werten war zu weit: unter Linux wirft `Paths.get` auch für einen Nicht-ASCII-Namen in einer JVM mit POSIX-Locale, obwohl die Datei existiert. Seit F238 gilt das nur noch für das NUL-Zeichen und für Namen, die die JVM kodieren kann (`OfflineModelGuardTest`); der Test oben bleibt grün.

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
- **Commit:** java-llama.cpp `6853464`

<a id="f59"></a>
### F59 – java-llama.cpp: `FfmNativeServer.isRunning()` kann nach einem parallelen `close()` freigegebenen Speicher lesen
- **Zeit:** 2026-10-17 21:25 UTC
- **Datei:** `llama-ffm/src/main/java/net/ladenthin/llama/ffm/FfmNativeServer.java` (`isRunning`, `close`)
- **Schwere:** niedrig
- **Beschreibung:** `isRunning()` prüft `closed` und ruft dann `jllama_server_is_running(handle)`. `close()` setzt `closed` und ruft `jllama_server_stop`, das laut `jllama_api.h` den Handle freigibt. Läuft `close()` (etwa aus einem Shutdown-Hook) zwischen der Prüfung und dem nativen Aufruf eines anderen Threads, liest `jllama_server_is_running` einen freigegebenen `jllama_server` – undefiniertes Verhalten im nativen Code, schlimmstenfalls ein JVM-Absturz. Der JNI-Zwilling `server.NativeServer` hält genau deshalb `isRunning()` und `close()` unter einem gemeinsamen `lifecycleLock`; die FFM-Klasse dokumentiert keine Einschränkung. (`FfmLlamaModel` hat dasselbe Muster, schreibt aber ausdrücklich, dass `close()` keinen neuen Aufruf überholen darf – dort ein dokumentierter Vertrag, kein Fund.)
- **Messung (J7, `BUGHUNT-MEASUREMENTS.md` in `4c58e8f`, JDK 25):** mit einer Naht rot – `close()` gab den Handle frei, während `isRunning()` noch im nativen Aufruf war; ohne Naht 862 Mio. Aufrufe in 30 Runden ohne sichtbaren Schaden (das Fenster ist schmal).
- **Fix:** Entscheidung des Nutzers: das Lock aus `NativeServer`. `isRunning()` und `close()` laufen unter `lifecycleLock`, `closed` ist ein einfaches Feld unter diesem Lock. Die beiden nativen Aufrufe liegen hinter der paketinternen Naht `FfmNativeServer.Backend` (wie `NativeServer.Backend`), statt veränderlicher statischer Method-Handles.
- **Tests:** `FfmNativeServerTest` (4, ohne Bibliothek): `close()` wartet auf ein `isRunning()` im nativen Aufruf und gibt erst danach frei; `isRunning()` nach `close()` fragt den Handle nicht ab; zweites `close()` ohne Wirkung; `isRunning()` fragt im offenen Zustand.
- **Rot-Grün-Nachweis:** lokal mit JDK 25 (Temurin 25.0.1, Toolchain): ohne Lock (Naht beibehalten, `AtomicBoolean` wie vorher) rot (1 von 4, `close()` wartete nicht), mit Lock grün.
- **Nachtrag (Branch-Review 2026-10-20):** Dieselbe Prüfen-dann-Rufen-Lücke hatte `FfmTextToSpeech`; dort mit demselben Lock als F198 behoben.
- **Status:** gefixt
- **Commit:** bd479b3

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
- **Commit:** java-llama.cpp `29ba25c`
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
- **Commit:** java-llama.cpp `196e76f`
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
- **Commit:** java-llama.cpp `980c48d` (F94)

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
- **Commit:** java-llama.cpp `fffb318`

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
- **Commit:** java-llama.cpp `39b008b`

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
- **Commit:** java-llama.cpp `9958f31`

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
- **Commit:** cb8a3ea
- **Nachtrag 2026-10-19 (Verhalten geändert):** J9 hat F67 bestätigt (`4c58e8f`: nach `setLoraAdapter(0, 0f)` lieferte der warme Slot „. He saw a picture of a bird …“ statt „ was a little girl …“ wie ein geleerter Slot). Entscheidung des Nutzers: upstream-fähiger Patch `0021` statt eines Leerens in `engine::set_lora_adapters`, weil upstreams eigene HTTP-Route `POST /lora-adapters` denselben Fehler hat. Der `else`-Zweig von `launch_slot_with_task` wendet jetzt dieselbe Regel an wie LoRA pro Anfrage (`lora_should_clear_cache`, aLoRA behält den Cache). Das deckt auch den zweiten Weg ab, den die Durchsicht fand: eine Anfrage mit eigenen Adaptern füllt den Cache, die nächste ohne läuft mit den Server-Adaptern darauf weiter. Nicht abgedeckt, in Javadoc/README genannt: der Host-RAM-Prompt-Cache (`--cache-ram`, standardmäßig an) speichert Slot-Zustände ohne ihre Adapter. Javadoc `setLoraAdapters`, README, CHANGELOG, CLAUDE.md (Patch- und Testtabelle), `docs/patches.md`. Upstream-master bewusst nicht geprüft (Entscheidung des Nutzers: zeigt sich beim nächsten Bump über den Drop-Check).
- **Tests (Patch):** `src/test/cpp/test_lora_cache.cpp` (4): Zufallsmodell + Zufalls-LoRA auf K, V und FFN-Ausgang, ein Slot, `--cache-ram 0`, greedy; nach Skala 1→0, 0→1 und nach einer Anfrage mit eigenen Adaptern muss der warme Slot wie ein geleerter antworten; ein vierter Fall schließt das leere Bestehen aus (Adapter ändert die Antwort, Wiederholung nutzt den Cache). Das Modellschreiben liegt jetzt in `tiny_llama_gguf.hpp`, gemeinsam mit `test_c_api.cpp`.
- **Rot-Grün-Nachweis (Patch):** ohne Patch rot (3 von 4; z. B. „mYmq5*mq“ statt „C+oM~kM~“), mit Patch grün (4 von 4, drei Läufe); ganze C++-Suite 633/633; alle 21 Patches wenden sich der Reihe nach sauber auf b11639 an; `check-patches.py` 0 Probleme.
- **Commit (Patch):** java-llama.cpp `e5a3efc`
- **Nachtrag (Branch-Review 2026-10-20):** Der Schutz gegen das leere Bestehen prüfte `tokens_cached > 0`; das ist bei b11639 `slot.prompt.n_tokens()`, also jedes Token, das der Slot nach der Anfrage hält, und nach jeder Anfrage positiv, auch ohne ein einziges wiederverwendetes (Fund J12-1). Jetzt `timings.cache_n > 0` (upstreams `n_prompt_cached` = `n_past`) und `timings.prompt_n < tokens_evaluated`; die drei Wechselfälle verlangen zusätzlich `timings.cache_n == 0` für die Anfrage nach dem Wechsel, sind damit ohne den Patch rot, auch wo das Zufallsmodell zufällig gleich antwortet. Gegenprobe: mit `"cache_prompt": false` für die Wiederholung ist der neue Schutz rot (`Expected: (cache_n(again)) > (0), actual: 0 vs 0`, `prompt_n` 77 = `tokens_evaluated` 77), der alte grün; ohne den `else`-Hunk von `0021` sind die drei Fälle rot (`cache_n(after_change) Which is: 76`, dazu die Inhaltsprüfung); mit allem 5/5 grün. Beim Prüfen bemerkt (offen, nicht Teil dieses Nachtrags): `LoraCacheFreshSlot` schreibt `slot0.bin` ins Arbeitsverzeichnis des Testlaufs statt in sein `--slot-save-path`, weil die Slot-API einen Pfad nimmt (`engine::slot_action` setzt `filepath = filename`, wie `LlamaModel.saveSlot(int, String filepath)` es dokumentiert); die Datei bleibt dort liegen.

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
- **Commit:** java-llama.cpp `70833e5`

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
- **Commit:** cb8a3ea

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
- **Commit:** java-llama.cpp `657e792`

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
- **Commit:** java-llama.cpp `4867189`

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
- **Commit:** java-llama.cpp `5a0b834`

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
- **Commit:** java-llama.cpp `b641834`

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
- **Commit:** java-llama.cpp `eeba74d`

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
- **Commit:** java-llama.cpp `262b41d`
- **Nachtrag (Branch-Review 2026-10-20, Fund J4-5):** Die Vorgabe 10000 und die Wirkungslosigkeit anderer Typen hatten keinen Test (nur `enabled` mit 1024). `AnthropicApiSupportTest#enabledThinkingWithoutBudgetUsesUpstreamsDefault` pinnt 10000, `#disabledThinkingSetsNoBudget` (`type` `disabled` mit `budget_tokens` 1024) das Fehlen von `thinking_budget_tokens`. Gegenprobe rot: Vorgabe 0 statt 10000 („Expected: is <10000> but: was <0>“), Budget für jeden `thinking`-Block statt nur `enabled` („Expected: is <false> but: was <true>“); mit dem Code grün (`AnthropicApiSupportTest` 36/36).

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
- **Commit:** java-llama.cpp `f6057ad`

<a id="f78"></a>
### F78 – java-llama.cpp: Ollama-`/api/chat` ignoriert `think` – `"think": false` schaltet das Denken nicht ab
- **Zeit:** 2026-10-18 02:30 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OllamaApiSupport.java:143` (`toOpenAiChatRequest`)
- **Schwere:** mittel
- **Beschreibung:** Ollamas Chat-API steuert das Denken eines Modells mit dem Top-Level-Feld `think`: `false` schaltet es ab, `true` an, ein String wählt bei gpt-oss eine Stufe (`"low"`, `"medium"`, `"high"`). `OllamaApiSupport.toOpenAiChatRequest` las das Feld nicht. Ein Ollama-Client, der für schnelle Antworten `think: false` schickt (Open WebUI, ollama-python), bekam trotzdem einen vollen Denkblock samt seiner Token und Latenz. Gefunden in der Musterdurchsicht nach F77.
- **Fix:** `think: false` wird zu `reasoning_effort: "none"`, ein nicht leerer String zu `reasoning_effort: <Stufe>`; `true` bleibt bei der Vorgabe der Vorlage (**korrigiert in F217:** `true` wird zu `chat_template_kwargs.enable_thinking: true`, sonst überging ein Server mit abgeschaltetem Denken die Anfrage still). Der Chat-Mapper wendet `reasoning_effort` seit F77 nach upstreams Regeln an (`enable_thinking: false` bzw. Template-Variable).
- **Reproduktion:** `OllamaApiSupportTest#toOpenAiChatRequestHonoursThinkLikeOllama` (Kette Übersetzer → `OpenAiRequestMapper` → `InferenceParameters.toJson()`)
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: is <false> but: was <true>“ – `enable_thinking` fehlte), mit Fix grün (24/24 der Klasse, skipped 0); Gegenprobe per `git stash` rot, nach `pop` grün. `OpenAiRequestMapperTest` 21/21, `OpenAiCompatServerHttpTest` 45/45, `LlamaArchitectureTest` 13/13, `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `6374dc8`

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
- **Commit:** java-llama.cpp `4a40a1d`

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
- **Commit:** java-llama.cpp `fdf21f1`

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
- **Commit:** java-llama.cpp `7fdc684`
- **Nachtrag (Branch-Review 2026-10-20, Fund J4-2):** Die zweite Hälfte des Fixes („oder der Stream endet“, das `closeThinkingBlock` in `end()`) hatte keinen Test: der Test oben schließt den Block über den Text. `AnthropicStreamTranslatorTest#reasoningFollowedByAToolCallIsClosedBeforeTheToolUseBlock` (Denken, dann nur ein Werkzeugaufruf: `signature_delta` < `content_block_stop` von Block 0 < `tool_use`-Block mit Index 1, `stop_reason` `tool_use`) und `#aThinkingOnlyAnswerIsClosedAtTheEnd` (nur Denken: `end()` liefert `signature_delta` und den Stopp von Block 0 vor `message_delta`, öffnet keinen weiteren Block) pinnen ihn. Gegenprobe ohne das `closeThinkingBlock` in `end()` rot („signature_delta is emitted / Expected: is <true> but: was <false>“, ebenso „end() emits the signature_delta“), mit ihm grün (`AnthropicStreamTranslatorTest` 11/11).
- **Nachtrag (Branch-Review 2026-10-20, Fund J4-6):** Die Javadoc war hinter F80/F81 zurückgeblieben: die Klassen-Javadoc von `AnthropicApiSupport` nannte nur `text`-/`tool_use`-/`tool_result`-Blöcke und Antworten aus `text` + `tool_use`, die von `onChunk`/`end` nur den Textblock. Seit Commit `368baf5` nennt die Klassen-Javadoc `image`- und `thinking`-Blöcke (eine Nachricht mit Bild bleibt als `text`/`image_url`-Teile, auch in einem `tool_result`; `thinking` wird `reasoning_content`, `reasoning_content` in der Antwort ein führender `thinking`-Block mit leerer Signatur), `onChunk` sagt, dass `reasoning_content` einen thinking-Block öffnet (einen offenen Textblock vorher schließt) und der erste nicht leere Text ihn mit leerem `signature_delta` + `content_block_stop` schließt, `end()`, dass es einen noch offenen thinking-Block schließt. Reine Doku, kein Test.

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
- **Commit:** java-llama.cpp `935342c`

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
- **Commit:** java-llama.cpp `f327156`
- **Nachtrag (Branch-Review 2026-10-20, Fund J5-6):** `output_index`, die Fortlaufendheit der `sequence_number` und das Ereignis `response.reasoning_text.done` hatten keinen Test; die Clients ordnen die Items nach `output_index` (das OpenAI-SDK wirft bei einem doppelten). `ResponsesStreamTranslatorTest#reasoningTextAndToolCallGetConsecutiveOutputIndicesAndSequenceNumbers` (begin, Denken, Text, Werkzeugaufruf, end) zerlegt die `data:`-Zeilen und verlangt genau 15 Ereignisse mit `sequence_number` 0..14, `output_item.added`/`.done` für `reasoning`/`message`/`function_call` an den Indizes 0/1/2, `response.reasoning_text.done` mit `item_id` `rs_resp_1`, `content_index` 0, vollem Text und `sequence_number` 6 sowie die Reihenfolge der Items in `response.completed`. Gegenprobe rot: Nachricht ohne eigenen Index („no response.output_item.added at output_index 2“), `sequence = 0` am Anfang von `end()` („Expected: is <[0, 1, …, 14]> but: was <[0, 1, 2, 3, 4, 5, 0, 1, …, 8]>“), ohne `reasoning_text.done` („Expected: is <15> but: was <14>“); mit dem Code grün (9/9). upstream (b11639) gibt weder `output_index` noch `sequence_number` aus; der Maßstab ist die OpenAI-Form.

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
- **Commit:** java-llama.cpp `638036f`

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
- **Commit:** java-llama.cpp `ad15d14`

<a id="f86"></a>
### F86 – java-llama.cpp: Responses-Werkzeugergebnis als Teil-Array kommt leer beim Modell an
- **Zeit:** 2026-10-18 03:50 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/ResponsesApiSupport.java:203` (`toolOutputParts`, neu; Aufruf in `appendInput`)
- **Schwere:** mittel
- **Beschreibung:** Ein `function_call_output` darf sein `output` als String oder als Array aus `input_text`-/`input_image`-Teilen schicken (z. B. ein Werkzeug, das einen Screenshot liefert). `appendInput` las `output.asText("")`; für ein Array ist das `""`, das Werkzeugergebnis erreichte das Modell also leer und es arbeitete ohne Ergebnis weiter. upstreams `convert_responses_to_chatcmpl` (`server-chat.cpp:189`–`220`, b11639) macht aus den Teilen `text`- und `image_url`-Teile der `role: tool`-Nachricht.
- **Fix:** ein Array-`output` wird zu Inhaltsteilen (`input_text` → `text`, `input_image` → `image_url`), andere Teiltypen werden übersprungen (upstream weist sie ab); ein String bleibt wie bisher.
- **Reproduktion:** `ResponsesApiSupportTest#functionCallOutputArrayKeepsItsPartsLikeUpstream`
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: is "text" but: was ""“; die erste Fassung des Tests lief über `get(0)` in eine NPE und wurde auf `path(0)` umgestellt, damit der Fehler an der Assertion sichtbar wird), mit Fix grün (16/16 der Klasse, skipped 0); Gegenprobe per `git stash` rot, nach `pop` grün. `ResponsesStreamTranslatorTest` 5/5, `OpenAiRequestMapperTest` 22/22, `OpenAiCompatServerHttpTest` 45/45, `LlamaArchitectureTest` 13/13, `spotbugs:check` grün.
- **Nachtrag (Branch-Review 2026-10-20):** Andere Teiltypen, ein Teil ohne `type` und ein Bild ohne URL werden seit F224 wie upstream mit 400 abgewiesen statt übersprungen; die Objektform `{url}` wird wie im Nachrichtenzweig gelesen.
- **Status:** gefixt
- **Commit:** java-llama.cpp `a2f8de1`

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
- **Commit:** java-llama.cpp `a4ac92a`

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
- **Commit:** java-llama.cpp `3ddf132`

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
- **Commit:** java-llama.cpp `576a886`

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
- **Commit:** java-llama.cpp `70779b3`
- **Nachtrag (Branch-Review 2026-10-20, Fund J4-5):** Die Wächter („nur wenn nach den fünf Zeichen ein `;` folgt“) hatten keinen Test; der Test oben nimmt nur wohlgeformte Kopfzeilen. `AnthropicApiSupportTest#aSystemTextThatIsNoWellFormedBillingHeaderIsLeftAlone` gibt `normalizeBillingHeader` einen Text ohne Kopfzeile, eine Kopfzeile ohne `cch=`, einen Stempel am Textende und einen Stempel ohne folgendes `;` und verlangt jeden unverändert zurück. Gegenprobe rot: ohne `end >= length` („StringIndexOutOfBounds Index 56 out of bounds for length 56“), ohne `charAt(end) != ';'` („Expected: is "x-anthropic-billing-header: cch=a5145 You are Claude Code." but: was "x-anthropic-billing-header: cch=fffff You are Claude Code."“); der Wächter `stamp < 0` ist eine äquivalente Mutante (ohne ihn liegt `end` im festen Präfix, dessen Zeichen dort kein `;` ist), kein Test kann ihn unterscheiden. Mit dem Code grün (`AnthropicApiSupportTest` 36/36). Wie upstream bleibt der Text still unverändert; upstream protokolliert zusätzlich `LOG_ERR`.

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
- **Commit:** java-llama.cpp `8b0872b`

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
- **Commit:** java-llama.cpp `7a64dd9`

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
- **Nachtrag (Branch-Review 2026-10-20):** Der Fix prüfte nur die Anfrageseite; kein Test fütterte den Usage-Chunk in der Form, die upstream mit `include_usage` sendet – ein eigener letzter Chunk mit leerem `choices`-Array, `usage` und `timings` nach dem Ende-Chunk (Fund J14-3). `StreamingChunkAssemblerTest#usageChunkWithEmptyChoicesCarriesTokenUsage` füttert genau die b11639-Folge (Inhalt, Ende-Chunk mit `finish_reason` und leerem `delta`, Usage-Chunk mit `id`/`model`/`object`/`system_fingerprint`) und pinnt 7/3 Token, das Teilstück `ok` und dass der Endegrund den Chunk ohne Wahl übersteht. Endegrund ist `length`, nicht `stop` wie im Vorschlag: ein fehlender Endegrund wird auf `STOP` abgebildet, mit `stop` bliebe ein Zurücksetzen unsichtbar. Ein reiner Absicherungstest ohne Verhaltensänderung, rot nur unter Mutation: `chunk.get("choices").get(0)` vor dem Lesen der Nutzung bzw. ein frühes `return` bei leerem `choices` (`Expected: is not null` auf `tokenUsage()`; ohne die bei der Prüfung ergänzte `notNullValue()`-Assertion war es eine `NullPointerException`), ein Zurücksetzen des Endegrunds auf einem Chunk ohne Wahl (`Expected: is <LENGTH> but: was <STOP>`); unverändert 13/13 grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `6c2adf8`

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
- **Commit:** java-llama.cpp `980c48d`

<a id="f95"></a>
### F95 – java-llama.cpp: gestreamtes `/v1/completions` und der Chat-Pfad verwerfen llama.cpps Sampling-Felder
- **Zeit:** 2026-10-18 05:30 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiRequestMapper.java:200` (`applyLlamaCppSampling`, neu; `readMaxTokens`)
- **Schwere:** mittel
- **Beschreibung:** Aus dem offenen Verdacht vom 2026-10-18 (gestreamtes `/v1/completions`). Das nicht gestreamte `/v1/completions` reicht den Körper an upstreams Schema durch (`server-schema.cpp`, b11639), das gestreamte baut `InferenceParameters` im `OpenAiRequestMapper` aus einer festen Feldliste. Alles außerhalb dieser Liste ging verloren: `min_p`, `typical_p`, `repeat_penalty`, `repeat_last_n`, `top_n_sigma`, `xtc_*`, `dynatemp_*`, `min_keep`, `n_keep`, `ignore_eos`, `grammar`/`json_schema`, und `n_predict` (upstreams Name, `max_tokens`/`max_completion_tokens` sind seine Aliase und verlieren gegen ihn). Dieselbe Anfrage erzeugte gestreamt also andere Ausgaben, eine Grammatik wirkte nur ohne `stream`. Derselbe Mapper bedient beide Chat-Pfade, die Felder fehlten dort ebenso. Alle Felder haben bereits einen Setter; anders als im Verdacht befürchtet, braucht dieser Teil keinen API-Umbau.
- **Fix:** `applyLlamaCppSampling` (aus `applyCommonFields`) übernimmt die Felder mit den vorhandenen Settern; `json_schema` nur ohne `grammar`, wie upstream (**korrigiert in F218:** das gilt nur für upstreams Completion-Schema; der Chat-Parser, den der Mapper seit F186 allein bedient, weist das Paar mit 400 ab, also werden beide Felder durchgereicht). `readMaxTokens` liest zuerst `n_predict`. Weiter offen (Verdacht bleibt): `logit_bias`, `n_probs`/`logprobs` (im gestreamten Completion-Chunk ohnehin nicht ausgegeben) und `n`.
- **Reproduktion:** `OpenAiRequestMapperTest#llamaCppSamplingExtensionsReachTheStreamedCompletion`, `#nPredictWinsOverMaxTokensLikeUpstream`, `#jsonSchemaReachesTheStreamedCompletionUnlessAGrammarIsGiven`
- **Rot-Grün-Nachweis:** ohne Fix rot (3 Fehlschläge: „Expected: is a numeric value within <1.0E-4> of <0.05> but: <0.0>“, „Expected: is <7> but: was <50>“, „Expected: is "object" but: was ""“), mit Fix grün (25/25 der Klasse, skipped 0); Gegenprobe per `git stash` rot (3), nach `pop` grün. `OpenAiCompatServerHttpTest` 46/46, `AnthropicApiSupportTest` 19/19, `ResponsesApiSupportTest` 18/18, `OllamaApiSupportTest` 24/24, `LlamaArchitectureTest` 13/13, `spotless:apply` und `spotbugs:check` grün (eine erste Fassung mit allen Feldern in einer Methode meldete `CE_CLASS_ENVY`, eine Map-Tabelle `DMC_DUBIOUS_MAP_COLLECTION`; jetzt je Feld ein Aufruf über `withFloat`/`withInt`).
- **Nachtrag (Branch-Review 2026-10-20):** Das hier „weiter offen“ Genannte ist erledigt: `logit_bias` und `n_probs` mit F180, das gestreamte `/v1/completions` mit F186 (upstreams eigener Stream, Wahrscheinlichkeiten eingeschlossen), `n > 1` mit F180 und auf allen nativen Wegen mit F197.
- **Status:** gefixt
- **Commit:** java-llama.cpp `1149533`

<a id="f96"></a>
### F96 – java-llama.cpp: Anthropic-Anfrage ohne `max_tokens` läuft unbegrenzt, `chat_template_kwargs` geht verloren
- **Zeit:** 2026-10-18 05:45 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/AnthropicApiSupport.java:111` (`toOpenAiChatRequest`)
- **Schwere:** niedrig
- **Beschreibung:** Aus dem offenen Verdacht „Anthropic-Rest nach F76“. upstreams `convert_anthropic_to_oai` (`server-chat.cpp:585`, b11639) setzt `max_tokens` 4096, wenn die Anfrage keines trägt, und reicht `chat_template_kwargs` durch. `/v1/messages` übernahm `max_tokens` nur, wenn vorhanden, eine Anfrage ohne generierte also, bis der Kontext voll war; und `chat_template_kwargs` (etwa Qwen3s `enable_thinking`) wurde verworfen, obwohl der nachgeschaltete `OpenAiRequestMapper` es seit F77 liest.
- **Fix:** Ohne Zahl in `max_tokens` wird `DEFAULT_MAX_TOKENS` (4096) gesetzt; ein Objekt in `chat_template_kwargs` wird kopiert.
- **Reproduktion:** `AnthropicApiSupportTest#requestWithoutMaxTokensIsCappedLikeUpstream`, `#requestPassesChatTemplateKwargsLikeUpstream` (zunächst ein Sammeltest, nach der Regel „ein Test je Richtung“ aufgeteilt)
- **Rot-Grün-Nachweis:** gegen die Produktionsdatei vor `4b20c57` rot, jeder Teil für sich (2 Fehlschläge: „Expected: is <4096> but: was <0>“, „Expected: is <true> but: was <false>“ für `chat_template_kwargs`), mit Fix grün (21/21 der Klasse, skipped 0). `OpenAiCompatServerHttpTest` 46/46, `AnthropicStreamTranslatorTest` 5/5, `OpenAiRequestMapperTest` 25/25, `ResponsesApiSupportTest` 18/18, `OllamaApiSupportTest` 24/24, `LlamaArchitectureTest` 13/13, `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `4b20c57`

<a id="f97"></a>
### F97 – java-llama.cpp: Ollama-`/api/chat` verliert den Denkinhalt in beide Richtungen
- **Zeit:** 2026-10-18 06:00 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OllamaApiSupport.java:267` (`toOllamaChatResponse`, `toOllamaContentLine`, `toOpenAiMessage`)
- **Schwere:** niedrig
- **Beschreibung:** Aus dem offenen Verdacht „Ollama-Antworten tragen den Denkinhalt nicht“. Ollama legt den Denkinhalt eines Modells in `message.thinking` ab, blockierend und je Stream-Zeile, und Clients senden ihn im Verlauf zurück. `OllamaApiSupport` übernahm aus OpenAIs Antwort nur `content` und `tool_calls`; im Stream lieferte `toOllamaContentLine` für einen Chunk ohne Text `null`, Denk-Deltas fielen also ganz weg. Seit F78 kann ein Client mit `think` das Denken steuern, sah es aber nie. Im Verlauf blieb `thinking` als unbekannter Schlüssel stehen, den llama.cpps Parser ignoriert (`common/chat.cpp:422` liest `reasoning_content`), wie bei F84 für Anthropic.
- **Fix:** Blockierend und im Stream wird `reasoning_content` zu `message.thinking` (eine Denk-Zeile trägt `content` leer); im Verlauf wird `thinking` zu `reasoning_content`.
- **Reproduktion:** `OllamaApiSupportTest#blockingChatAnswerCarriesTheThinkingLikeOllama`, `#streamedThinkingDeltaBecomesAThinkingLineLikeOllama`, `#historyThinkingBecomesReasoningContent` (zunächst ein Sammeltest, nach der Regel „ein Test je Richtung“ aufgeteilt)
- **Rot-Grün-Nachweis:** gegen die Produktionsdatei vor `fb4dd64` rot, jede Richtung für sich (3 Fehlschläge: „Expected: is "6 times 7" but: was ""“ blockierend und im Verlauf, „Expected: is "6 ti" but: was ""“ im Stream), mit Fix grün (27/27 der Klasse, skipped 0). `OpenAiCompatServerHttpTest` 46/46, `AnthropicApiSupportTest` 20/20, `ResponsesApiSupportTest` 18/18, `OpenAiRequestMapperTest` 25/25, `LlamaArchitectureTest` 13/13, `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `fb4dd64`

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
- **Commit:** java-llama.cpp `95af953`

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
- **Commit:** java-llama.cpp `1939435`

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
- **Commit:** java-llama.cpp `57cbdc0`

<a id="f103"></a>
### F103 – java-llama.cpp: `max_tokens: 0` generiert unbegrenzt statt nur den Cache zu füllen
- **Zeit:** 2026-10-18 07:35 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiRequestMapper.java:164` (`applyCommonFields`)
- **Schwere:** niedrig
- **Beschreibung:** Musterdurchsicht nach F102 (Werte, die der Mapper anders behandelt als upstream). upstreams Schema (`server-schema.cpp`, `n_predict` mit den Aliasen `max_completion_tokens`/`max_tokens`, Grenzen -1..INT32_MAX, b11639): „When 0, no tokens will be generated but the prompt is evaluated into the cache“. Der Mapper übernahm die Obergrenze nur bei `> 0`; ein Client, der mit `0` den Cache vorwärmt, bekam auf den gemappten Wegen (beide Chat-Pfade, gestreamtes `/v1/completions`, und damit Anthropic/Responses/Ollama) eine Generierung ohne Obergrenze bis zum Kontextende, das wörtlich durchgereichte `/v1/completions` dagegen keine.
- **Fix:** Die Obergrenze wird bei `>= 0` gesetzt; `-1` bzw. kein Feld lässt upstreams Standard (unbegrenzt). (**Ergänzt in F219:** jeder Wert außer -1 wird durchgereicht, damit upstream -2 und darunter mit 400 abweist.)
- **Reproduktion:** `OpenAiRequestMapperTest#zeroTokenCapIsForwardedAsUpstreamReadsIt` (bis 2026-10-21 `zeroTokenCapOnlyWarmsTheCacheLikeUpstream`)
- **Rot-Grün-Nachweis:** ohne Fix rot („Expected: is <0> but: was <-1>“), mit Fix grün (27/27 der Klasse, skipped 0); Gegenprobe per `git stash` rot (1), nach `pop` grün. `OpenAiCompatServerHttpTest` 47/47, `AnthropicApiSupportTest` 21/21, `ResponsesApiSupportTest` 19/19, `OllamaApiSupportTest` 27/27, `LlamaArchitectureTest` 13/13, `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `d7f91db`
- **Nachtrag (Branch-Review 2026-10-20, Fund J2-2):** Titel und Beschreibung beschreiben eine Wirkung, die der Server nicht hat: `max_tokens: 0` füllt nicht nur den Cache, sondern erzeugt **ein** Token (Messung J2, F158; `withNPredict` sagt es seit dem Branch). Upstreams README-Satz („When 0, no tokens will be generated but the prompt is evaluated into the cache“) stammt aus PR #1937, dessen Zweig `n_predict == 0` → EOS es bei b11639 nicht mehr gibt; `server-context.cpp` übernimmt 0 unverändert in `n_predict_max`, und die einzige Budgetprüfung (`slot.stats.n_gen > 0 && … && !slot.has_budget()`) läuft erst nach dem ersten Token. Der Fix bleibt richtig (0 wird wie auf dem wörtlich durchgereichten `/v1/completions` weitergegeben statt „unbegrenzt“ zu bedeuten); der Mapper-Kommentar nennt das eine Token jetzt, der Test heißt `zeroTokenCapIsForwardedAsUpstreamReadsIt` mit unveränderter Assertion (`n_predict` ist 0). Reine Doku, kein neuer Test.

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
- **Commit:** java-llama.cpp `ce983cf`

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
- **Commit:** java-llama.cpp `f9e680a`

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
- **Commit:** java-llama.cpp `b370643`

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
- **Commit:** java-llama.cpp `d58662f`

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
- **Commit:** java-llama.cpp `8e91168`

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
- **Nachtrag (Branch-Review 2026-10-20, Funde J2-6, J5-8):** Erledigt. Die Klassen-Javadoc von `OpenAiRequestMapper` nennt als wörtlich weitergereicht nur noch `messages` und `tools`; `tool_choice` geht in der Textform weiter, ein benanntes Werkzeug wird zu `required` mit nur diesem Werkzeug (F227), jede andere Objektform (`allowed_tools`, `custom`) wird ignoriert und läuft als `auto`, `parallel_tool_calls` wird als Boolean gelesen. Der Kommentar über `tool_choice` in `ResponsesApiSupport.toOpenAiChatRequest` sagt dasselbe. Reine Doku, kein Test.
- **Nachtrag (Branch-Review 2026-10-20):** Mit mehreren Werkzeugen ließ `required` jeden Aufruf zu; seit F227 geht auf beiden Wegen nur das benannte Werkzeug weiter, ein unbekannter oder fehlender Name ist 400.
- **Status:** gefixt
- **Commit:** java-llama.cpp `c5823a5`

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
- **Commit:** java-llama.cpp `6ff8eea`
- **Nachtrag (Branch-Review 2026-10-20, Fund J6-2):** `toResponse` hatte drei Zweige, die kein Test sah: kein Test schickte eine `documents`-Anfrage durch `toResponse` (nur der modellgestützte `OpenAiServerRerankIntegrationTest`, der sich lokal ohne Reranker überspringt), keine TEI-Anfrage setzte `top_n`, und keine prüfte, dass `text` ohne `return_text` fehlt. `OaiRerankSupportTest#documentsRequestGetsTheOpenAiObjectThroughToResponse` pinnt die OpenAI-Form samt Modell-Id, `#teiRequestHonoursTopNAndOmitsTextByDefault` die TEI-Form mit `top_n: 1` (nur der beste Eintrag) und ohne `text` (upstreams `format_response_rerank`, b11639: `resize(min(top_n, n))` vor der TEI-Rückkehr, `return_text` standardmäßig `false`; das `top_n` betrifft auch F177). Gegenprobe rot: Bedingung `true` („Expected: is <true> but: was <false>“ auf `isObject()`), `readTopN(request)` → `-1` im TEI-Zweig („Expected: is <1> but: was <2>“), `asBoolean(false)` → `asBoolean(true)` („Expected: is <false> but: was <true>“); unverändert 24/24 der Klasse grün.
- **Nachtrag (Branch-Review 2026-10-20, Fund J6-3):** Die TEI-Form über die echte Route: `OpenAiServerRerankIntegrationTest#teiRequestIsAnsweredAsABareArray` schickt `texts` mit `return_text: true` und prüft das bloße Array mit `index`/`score` (kein `relevance_score`) und dass `text` das zum `index` gehörige gesendete Dokument ist. Lokal mit dem Winzmodell als Reranker grün; mit nicht erkanntem `texts` rot („Expected: is <true> but: was <false>“ auf `isArray()`).
- **Nachtrag (Branch-Review 2026-10-20, Fund J6-6):** Die README-Routentabelle, die Klassen-Javadoc von `OaiRerankSupport`, die Javadoc von `OpenAiBackend.rerank` und die `server`-Zeile in `CLAUDE.md` beschrieben `/v1/rerank` nur mit `results`/`data`. Seit Commit `c5e53cc` nennen alle vier die TEI-Anfrage (`texts`, `return_text`), das bloße Array `[{index, score[, text]}]`, die Wahl der Form nach `texts` wie upstreams `format_response_rerank` und das `usage`-Objekt der OpenAI-Form (F177). Reine Doku, kein Test.

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
- **Commit:** java-llama.cpp `909edc5`

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
- **Commit:** java-llama.cpp `b313ad2`

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
- **Commit:** java-llama.cpp `03ad149`

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
- **Commit:** java-llama.cpp `3e2f3b5`

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
- **Commit:** java-llama.cpp `e4f54a6`

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
- **Commit:** java-llama.cpp `bd3db6a`
- **Nachtrag (Branch-Review 2026-10-20, Fund J5-6):** Die laufende `sequence_number` des `response.failed` war nur über den Server und nur per Teilstring geprüft. `ResponsesStreamTranslatorTest#failedContinuesTheSequenceNumber` (begin, ein Textchunk, `failed("boom", "server_error")`) verlangt fünf Ereignisse, das letzte `response.failed` mit `sequence_number` 4, `status` `failed` und `error.code`/`error.message`. Gegenprobe mit `sequence = 0` in `failed()` rot („Expected: is <4> but: was <0>“), mit dem Code grün (`ResponsesStreamTranslatorTest` 9/9).

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
- **Commit:** java-llama.cpp `ae72c29`

<a id="f123"></a>
### F123 – java-llama.cpp: `logprobs`/`top_logprobs` einer Chat-Anfrage gehen verloren
- **Zeit:** 2026-10-18 14:35 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiRequestMapper.java:72` (`toInferenceParameters`)
- **Schwere:** niedrig
- **Beschreibung:** Eine OpenAI-Chat-Anfrage mit `"logprobs":true` (und optional `"top_logprobs":N`) bekam keine Token-Wahrscheinlichkeiten: Der Mapper übernahm keins der beiden Felder, `InferenceParameters` trug also kein `n_probs`. Upstream bildet sie in `oaicompat_chat_params_parse` (`server-common.cpp:1456`, b11639) auf `n_probs = top_logprobs` ab, Standard 20, und die nativen Chat-Ergebnisse tragen die Wahrscheinlichkeiten dann von selbst (blockierend und im OAI-Chunk-Pfad). Teil des offenen F95-Verdachts („`n_probs`/`logprobs`“), der als Blocker eine API-Erweiterung nannte – `withNProbs` gab es aber schon (Erkenntnis F95: Verdacht in Teile zerlegen). Gefunden per Probe-Klasse über den Mapper.
- **Fix:** Bei `logprobs: true` setzt der Chat-Mapper `withNProbs(top_logprobs)`, ohne `top_logprobs` 20 wie upstream. Nur im Chat-Pfad: Das `logprobs` des alten `/v1/completions` ist eine Zahl mit anderer Bedeutung. Nicht übernommen: upstreams zwei Ablehnungen (`top_logprobs` ohne `logprobs`; `logprobs` mit Werkzeugen im Stream). (**Korrigiert in F220:** die erste Ablehnung gilt jetzt mit upstreams Text, und ein ganzzahliges `logprobs` ist wie bei upstream der Alias von `n_probs` statt der Chat-Form; die zweite bleibt offen.)
- **Reproduktion:** `OpenAiRequestMapperTest#logprobsWithTopLogprobsBecomesNProbs`, `#logprobsWithoutTopLogprobsDefaultsToTwentyLikeUpstream` (dazu als Schutz `#logprobsFalseLeavesNProbsUnset`)
- **Rot-Grün-Nachweis:** ohne Fix beide rot („Expected: is <3> but: was <-1>“ bzw. „Expected: is <20> but: was <-1>“), mit Fix grün (31/31 der Klasse, skipped 0); Gegenprobe per `git stash` rot (2), nach `pop` grün. Breite Liste wie F117 grün, `spotless:apply` und `spotbugs:check` grün. Der erste Grün-Lauf scheiterte am Compiler (`NotJavadoc`: Konstante zwischen fremde Javadoc und Deklaration eingefügt, wie bei F108) und meldete trotzdem den alten roten Bericht – das Prüfskript löscht den Bericht jetzt auch vor dem Grün-Lauf.
- **Status:** gefixt
- **Commit:** java-llama.cpp `578614c`

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
- **Commit:** java-llama.cpp `72cf2ea`

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
- **Nachtrag (Branch-Review 2026-10-20):** Die beiden Tests oben prüften nur, dass nichts wirft, und die fertige Anfrage; welche Fragmente als `PartialToolCall` gemeldet werden und mit welcher Id, war ungeprüft (Fund J14-2). `#toolCallDeltaWithoutArguments_doesNotFailTheStream` pinnt jetzt genau ein gemeldetes Teilstück (Index 0, Id `call_1`, Name `get_weather`, Argumente `{}`), einen `CompleteToolCall` und langchain4js Vertrag über den neuen Helfer `assertPartialsMakeUpTheCompleteCall` (die Argumentstücke ergeben aneinandergehängt die Argumente des fertigen Aufrufs, jedes trägt dessen Id und Namen). `#toolCallDeltaBeforeItsName_doesNotFailTheStream` prüft nur, dass mindestens ein Teilstück gemeldet wird (`is(not(empty()))`, damit die Schleife nicht leer durchläuft; bei der Prüfung ergänzt, rot gegen eine Mutante, die nach einem zurückgehaltenen Fragment nichts mehr meldet: `Expected: is not an empty collection`) und dass jedes gemeldete Teilstück die endgültige Id und den Namen trägt: der Vertrag ist dort heute rot (`Expected: is "{\"city\":1}" but: was "ty\":1}"`, das Präfix vor dem Namen wird nie gemeldet, Fund J14-1) und kommt mit dessen Fix; das heutige einzelne Teilstück wird bewusst nicht festgeschrieben. Gegenprobe rot: die Id aus dem aktuellen Fragment statt aus dem gesammelten Aufruf (`Expected: is "call_1" but: was ""`), das Kopf-Fragment mit einem Platzhalter-Argument `" "` gemeldet (`Expected: is <1> but: was <2>`); unverändert 12/12 grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `f7f3f0f`
- **Nachtrag (Branch-Review 2026-10-20, Fund J14-4):** Die Klassen-Javadoc von `StreamingChunkAssembler` sagte noch, jedes `tool_calls`-Fragment gehe als `onPartialToolCall` hinaus, und dass id und Name am ersten Fragment kämen. Seit dem Fix (`4d599fa`) wird ein Fragment nur weitergereicht, wenn es Argumenttext trägt und sein Aufruf schon einen Namen hat; andere (ein Kopf nur mit id/Name, Argumente vor dem Namen) speisen nur den zusammengesetzten Aufruf. Javadoc und der Satz im `llama-langchain4j/README.md` sagen das jetzt, „id und Name“ kommen „usually“ am ersten Fragment. Nur Doku.

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
- **Commit:** java-llama.cpp `b4630d5`

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
- **Commit:** java-llama.cpp `5bba245`
- **Nachtrag (Branch-Review 2026-10-20, Fund J8-4):** Zwei Stellen in `JsonParameters` beschrieben noch das alte Verhalten: das `@param text` von `withOptionalJson` („`null` means leave the parameter unset“) und die Javadoc von `toJsonString` („null to behave as do not set this parameter“). Beide sagen jetzt, dass `null` den früheren Wert löscht, wie die Beschreibung, das `@return` und die Aufrufer in `InferenceParameters`. Nur Doku.

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
- **Commit:** java-llama.cpp `7f79298`

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
- **Commit:** java-llama.cpp `806571d`

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
- **Commit:** java-llama.cpp `017ac95`

<a id="f158"></a>
### F158 – java-llama.cpp: `ModelParameters.setPredict(-2)` verspricht „until context filled“, der Server erzeugt damit je Anfrage nur ein Token
- **Zeit:** 2026-10-19 00:10 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/parameters/ModelParameters.java:202` (Javadoc `setPredict`)
- **Schwere:** niedrig bis mittel (still: keine Fehlermeldung, nur abgeschnittene Antworten)
- **Beschreibung:** Gefunden beim Nachprüfen von F69. Upstreams `-n` (`common/arg.cpp:1661`, b11639) nennt „-2 = until context filled“ **nur** für `LLAMA_EXAMPLE_COMPLETION`, für den Server lautet die Beschreibung „-1 = infinity“; der Parser übernimmt jeden Wert. Im Server übernimmt ein Slot `params_base.n_predict`, wenn die Anfrage `-1` sendet oder das Feld weglässt (`server-context.cpp:1964`), und `has_budget()` (`:495`) zählt nur `-1` als unbegrenzt: bei `-2` ist `n_remaining()` negativ, und die Grenzprüfung (`:2040`, `n_gen > 0 && !has_budget()`) beendet die Erzeugung nach dem ersten Token mit `STOP_TYPE_LIMIT`. Aus dem Quelltext abgeleitet, **nicht gemessen**.
- **Fix:** Dokumentation korrigiert (ohne Test, Nutzerregel): Javadoc nennt nur noch `-1 = infinity` und erklärt, dass `-2` eine Einstellung des Completion-Tools ist und hier nach dem ersten Token stoppt. **Offen, Entscheidung des Nutzers:** `-2` in `ModelParameters.setPredict` mit `IllegalArgumentException` abweisen – modellfrei testbar, ändert aber das Verhalten eines öffentlichen Setters.
- **Messung (J1, `BUGHUNT-MEASUREMENTS.md` in `4c58e8f`):** bestätigt – `-2`: 1 Token, `stop_type="limit"`; `-1`: 2042 Tokens (bis Kontextende); `16`: 16. Dazu J2: `withNPredict(0)` erzeugt ebenfalls 1 Token, nicht 0.
- **Nachtrag (Entscheidung des Nutzers nach der Messung):** `setPredict` lehnt Werte unter `-1` mit `IllegalArgumentException` ab; Javadoc von `setPredict` und `withNPredict` sagt „`0` verhält sich wie `1`“. Test `ModelParametersExtendedTest#testSetPredictBelowMinusOneIsRejected` (`-2`, `-3`, `Integer.MIN_VALUE`) ersetzt `testSetPredictFillContext`, dazu `testSetPredictZero`.
- **Rot-Grün-Nachweis:** ohne die Prüfung rot (3 von 3), mit ihr grün (`ModelParametersExtendedTest`, `ModelParametersTest`, `WireNameRegistryTest`: 252 Tests).
- **Nachtrag (Branch-Review 2026-10-20):** Die Übersicht nannte bis zum Branch-Review noch „Doku gefixt, Verhalten offen“; das Verhalten ist seit `be4d4a2` geändert (siehe oben).
- **Status:** gefixt
- **Commit:** cb8a3ea (Doku), be4d4a2 (Verhalten)

<a id="f165"></a>
### F165 – java-llama.cpp: `OpenAiCompatServer` ohne `-c` bewirbt 8192 + 2048 Tokens statt des Fensters des geladenen Modells
- **Zeit:** 2026-10-19 14:40 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiServerCli.java` (`toServerConfig`), `OpenAiCompatServer.java` (`main`)
- **Schwere:** mittel (ein Client, der seinen Prompt nach `/props`/`/api/show` bemisst, schickt bis zu 5× zu viel Kontext)
- **Historie:** Die Budgets kamen nur aus `-c` (seit F70 als ganzes Slot-Fenster aufgeteilt); ohne `-c` blieben die festen Vorgaben stehen.
- **Messung (J3, `BUGHUNT-MEASUREMENTS.md` in `4c58e8f`):** Draft-Modell ohne `-c`: `/props` `n_ctx` 10240, `/api/show` `num_ctx 10240`, `/slots` `n_ctx` 2048 (= `n_ctx_train`).
- **Fix:** Entscheidung des Nutzers: das echte `n_ctx` bewerben. Neue Überladung `toServerConfig(supportsVision, modelFtype, slotContextSize)`; `main` übergibt `n_ctx` des ersten Slots aus `getMetricsTyped()`. Ein bekanntes Slot-Fenster geht vor `-c` (die Last kann kleiner ausfallen, etwa durch `--fit`), sonst gilt wie bisher `-c` bzw. die Vorgabe.
- **Tests:** `OpenAiServerCliTest#loadedSlotContextIsAdvertisedWithoutContextSize`, `#loadedSlotContextWinsOverTheContextSizeArgument`, `#unknownSlotContextFallsBackToTheContextSizeArgument`.
- **Rot-Grün-Nachweis:** mit der alten Auswahl (nur `-c`) rot (2 von 3, der Rückfalltest ist beidseitig grün), mit dem Fix grün (`OpenAiServerCliTest`, `OpenAiCompatServer*Test`: 114 Tests). Die Verdrahtung in `main` braucht ein Modell und ist hier nur übersetzt; Prüfung mit dem Befehl aus J3 (erwartet: 2048 in `/props` und `/api/show`).
- **Status:** gefixt
- **Commit:** 490f43b

<a id="f166"></a>
### F166 – java-llama.cpp: Leere Werkzeug-Argumente im Verlauf lassen die ganze Folgeanfrage mit 400 scheitern
- **Zeit:** 2026-10-19 15:00 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/parameters/ParameterJsonSerializer.java` (`buildMessages(List<ChatMessage>)`)
- **Schwere:** niedrig (mit dem gemessenen Modell nicht erreichbar, im Eintrittsfall aber Totalausfall der Folgeanfrage)
- **Historie:** `buildMessages` schreibt `ToolCall.getArgumentsJson()` seit den Werkzeugaufrufen im Verlauf wörtlich.
- **Messung (J4, `BUGHUNT-MEASUREMENTS.md` in `4c58e8f`):** Qwen2.5-1.5B liefert für ein Werkzeug ohne Parameter 20/20 `"{}"`, nie `""`; mit `""` im Verlauf antwortet der Server `InvalidRequestException: Failed to parse tool call arguments as JSON … attempting to parse an empty input`, mit `"{}"` normal.
- **Fix:** Entscheidung des Nutzers: leere (auch nur aus Leerraum bestehende) Argumente als `"{}"` schreiben, alles andere unverändert.
- **Tests:** `ParameterJsonSerializerTest#testBuildMessages_emptyToolCallArguments_areWrittenAsAnEmptyObject` (`""`, `" "`, `"\n"`), `#testBuildMessages_toolCallArguments_arePassedOnVerbatim`.
- **Rot-Grün-Nachweis:** ohne den Fix rot (3 von 3), mit ihm grün (186 Tests in den Serializer-, Multimodal-, `InferenceParameters`- und `ChatRequest`-Klassen).
- **Nicht abgedeckt (offen):** Die Übersetzungen des Servers bauen den Verlauf ohne `buildMessages`: `ResponsesApiSupport.java:218` (`asText("")`) und `AnthropicApiSupport.java:229` (`input` fehlt → `""`) können weiterhin `""` schreiben. **Nachtrag 2026-10-19:** behoben als F172.
- **Status:** gefixt (Java-API-Pfad; Server-Übersetzungen als F172)
- **Commit:** 7f19450
- **Nachtrag (Branch-Review 2026-10-20, Fund J8-6):** `ParameterJsonSerializer#buildMessages(List<ChatMessage>)`, `InferenceParameters#withMessages(List<ChatMessage>)` und `ChatRequest#buildMessagesJson` sagen jetzt, dass leere oder nur aus Leerraum bestehende Tool-Call-Argumente als `{}` geschrieben werden. Genauer als der Kommentar am Fix: upstream `json::parse`t die Argumente des Verlaufs nur bei Templates, die Objekt-Argumente nehmen (`supports_object_arguments`, `common/chat.cpp:1350`, b11639); der Code-Kommentar sagt das jetzt auch. `ToolCall` sagt, dass die Argumente aus der Modellantwort unverändert bleiben und nur beim Zurückschreiben in den Verlauf aus leer `{}` wird. Nur Doku; `ParameterJsonSerializerTest` pinnt das Verhalten.

<a id="f167"></a>
### F167 – java-llama.cpp: `completeBatch`/`chatBatch` laufen auf dem gemeinsamen `ForkJoinPool` und nutzen auf wenigen Kernen nicht alle Slots
- **Zeit:** 2026-10-19 15:30 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/LlamaModel.java` (`completeBatch`, `completeBatchWithStats`, `chatBatch`), neu `BatchDispatch.java`
- **Schwere:** niedrig bis mittel (Durchsatz; die Javadoc versprach die `setParallel`-Slots)
- **Historie:** Jede Anfrage lief über `CompletableFuture.supplyAsync`, also auf dem gemeinsamen Pool mit `availableProcessors() - 1` Threads, den sie zudem mit der Anwendung teilte.
- **Messung (J5, `BUGHUNT-MEASUREMENTS.md` in `4c58e8f`):** 8 Slots, 8 Anfragen: 16 Kerne → 8 gleichzeitig (3225 ms); 4 Kerne → 3 (14067 ms); 16 Kerne mit Pool-Parallelität 3 → 3.
- **Fix:** Entscheidung des Nutzers: `completeBatch`/`completeBatchWithStats` übergeben alle Anfragen im Aufrufer-Thread (`requestCompletion`) und empfangen dann der Reihe nach; scheitert eine Übergabe, werden die bereits übergebenen freigegeben (`releaseTask`). Für `chatBatch` gibt es keine Trennung von Übergabe und Warten (`handleChatCompletions` wartet in einem nativen Aufruf), daher läuft dort jede Anfrage auf einem eigenen Daemon-Thread eines Executors, der genau so groß ist wie der Batch und mit dem Aufruf endet. Beides in `BatchDispatch` (ohne native Bibliothek testbar, `LlamaModel` lädt sie im statischen Initialisierer). Die SpotBugs-Ausnahme für den bewussten Rethrow ist von `LlamaModel` dorthin gezogen, zwei neue mit Begründung (`HES_LOCAL_EXECUTOR_SERVICE`, `ITC_INHERITANCE_TYPE_CHECKING`).
- **Tests:** `BatchDispatchTest` (8): alle Übergaben vor dem ersten Empfang, Übergabe im Aufrufer-Thread, Freigabe nach gescheiterter Übergabe, Empfang aller trotz Fehler, mehr gleichzeitige Anfragen als der gemeinsame Pool Threads hat (Latch über `commonPool().getParallelism() + 2`), Reihenfolge, Fehler nach den anderen, leere Eingabe.
- **Rot-Grün-Nachweis:** mit dem alten Mechanismus (`supplyAsync` je Anfrage, keine Freigabe) rot (4 von 8; die übrigen 4 prüfen unverändertes Verhalten), mit dem Fix grün. Ein modellgestützter Lauf (J5-Probe) steht aus: erwartet `maxProcessingSlots=8` auch auf 4 Kernen für `completeBatch`.
- **Status:** gefixt
- **Commit:** 8130463
- **Nachtrag (Branch-Review 2026-10-20):** `submitAllThenReceive` fing nur `RuntimeException`; ein `Error` übersprang Freigabe und restliche Empfänge (behoben als F235). Spätere Fehler eines Batches gingen ohne Spur verloren (behoben als F236, gilt auch für `runEachOnItsOwnThread`).

<a id="f168"></a>
### F168 – java-llama.cpp: `JllamaNative.takeStrings` gibt ein leeres Nicht-NULL-Array nicht frei
- **Zeit:** 2026-10-19 16:30 UTC
- **Datei:** `llama-ffm/src/main/java/net/ladenthin/llama/ffm/JllamaNative.java` (`takeStrings`)
- **Schwere:** niedrig (8 Byte je Aufruf mit leerem Ergebnis, praktisch unerreichbar)
- **Historie:** Die Prüfung `pointer == NULL || count == 0` vor dem `try` stammt aus der ersten Fassung der FFM-Bindungen; `jllama_rpc_server_devices` legt auch für 0 Einträge ein Array an.
- **Messung (J8, `BUGHUNT-MEASUREMENTS.md` in `4c58e8f`):** ein absichtlich falsch ausgerichteter Nicht-NULL-Zeiger mit Anzahl 0: `jllama_strings_free` direkt bricht mit `free(): invalid pointer` ab (Exit 134), über `takeStrings` überlebt die JVM – die Freigabe wurde nie gerufen.
- **Fix:** Entscheidung des Nutzers: bei Nicht-NULL immer freigeben. Nur noch `NULL` kehrt früh zurück; die Freigabefunktion ist über eine paketinterne Überladung `takeStrings(out, length, free)` einsetzbar.
- **Tests:** `JllamaNativeTest#takeStringsFreesANonNullArrayEvenWhenItIsEmpty`, `#takeStringsDoesNotFreeANullArray`.
- **Rot-Grün-Nachweis:** lokal mit JDK 25 (Toolchain) und einem selbst gebauten `libjllama.so` (b11639, statisch): mit der alten Prüfung rot (1 von 9, der neue Test), mit dem Fix grün; `mvn -f llama-ffm/pom.xml verify` grün (29 Tests, 16 modellgestützte übersprungen, kein Modell).
- **Nachtrag (Branch-Review 2026-10-20):** Die beiden Tests oben setzen die Anzahl auf 0 und erreichen weder die Leseschleife noch die Anzahl, mit der ein gefülltes Array freigegeben wird; der einzige Test mit Einträgen (`rpcServerDevicesResolveTheCpuAndRefuseAnUnknownName`) hat genau einen, sieht also keine Reihenfolge, und eine falsche Anzahl leckt nur innerhalb von `jllama_strings_free` (Fund J13-2). Neu, als Charakterisierung des unveränderten Verhaltens: `JllamaNativeTest#takeStringsReadsEveryEntryInOrderAndFreesTheArrayWithItsCount` (zwei Arena-Strings, aufzeichnende Freigabe; `["alpha", "beta"]` und `<Adresse>/2`). Gegenprobe rot: Freigabe mit 0 (`Expected: is <[…/2]> but: was <[…/0]>`), rückwärts gelesen (`Expected: is <[alpha, beta]> but: was <[beta, alpha]>`); unverändert 10/10 in der Klasse grün.
- **Status:** gefixt
- **Commit:** 631d0ab

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
- **Nachtrag (Branch-Review 2026-10-20):** Auf der Responses-Route wurden auch Objekt-`arguments` zu `"{}"` (Jacksons `asText` eines Containers ist `""`); seit F225 ist ein Wert, der weder fehlt noch `null` noch ein String ist, wie upstream eine 400.
- **Status:** gefixt
- **Commit:** java-llama.cpp `ce6e5a7`

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
- **Commit:** java-llama.cpp `2db54c9`

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
- **Commit:** java-llama.cpp `cb21ecf`

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
- **Commit:** java-llama.cpp `0dd5e41`

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
- **Commit:** java-llama.cpp `ef443b0`

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
- **Commit:** java-llama.cpp `68f81b6`
- **Nachtrag (Branch-Review 2026-10-20):** Die Abweisung eines negativen `top_n` griff nur für die ganzzahlige Schreibweise; ein `-1.5` galt als „fehlt“. Seit F230 liest `readTopN` jede Zahl, die Abweisung deckt auch die Gleitkomma-Schreibweise ab (`OaiRerankSupportTest#negativeFloatTopNIsAnInvalidRequest`).
- **Nachtrag (Branch-Review 2026-10-20, Fund J6-3):** Ende zu Ende nachgeholt, über die echte Route und `LlamaModelBackend.rerank`: `OpenAiServerRerankIntegrationTest#negativeTopNIsRejectedBeforeAnythingIsEvaluated` (400, `invalid_request_error`, Meldung nennt `top_n`, und `getMetricsTyped().getCumulativeProcessedPromptTokens()` bleibt gleich – der Status allein kann die Reihenfolge nicht zeigen, weil `toResponse` `top_n` nach dem nativen Aufruf noch einmal liest) und `#topNZeroAnswersNoResultsButUsageCountsTheEvaluatedTokens` (keine `results`/`data`, `usage.prompt_tokens > 0` und gleich `total_tokens` – der Beweis, dass das echte `server_task_result_rerank` `tokens_evaluated` in die Summe trägt; der C++-Test nutzt ein unechtes Ergebnis). Lokal gelaufen mit dem zufälligen 2-Layer-Llama als Reranker (`-Dnet.ladenthin.llama.rerank.model=<tiny.gguf>`, Rank-Pooling über den letzten Zustand), 4/4, nichts übersprungen; CI nimmt den Jina-Reranker. Gegenprobe rot: ohne den frühen `readTopN`-Aufruf im Backend („Expected: is <419L> but: was <493L>“ beim Zähler, Status weiter 400), mit dem Stand vor F177 in Java („Expected: is <400> but: was <200>“, „Expected: is <0> but: was <2>“), ohne `usage` („Expected: a value greater than <0> but: <-1> was less than <0>“).
- **Nachtrag (Branch-Review 2026-10-20, Fund J11-3):** Der Kommentar von `jllama_rerank` im ausgelieferten C-Header `jllama_api.h` nannte je Eintrag nur `index` und `score`; er nennt jetzt alle vier Felder, die `rerank_results_to_json` schreibt (`document`, `index` als Position in `documents`, `score`, `tokens_evaluated` mit 0, wenn upstream nichts meldet), wie die Javadoc von `LlamaModel#handleRerank`. Nur Doku.

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
- **Commit:** java-llama.cpp `887eb93`

<a id="f179"></a>
### F179 – java-llama.cpp: Ollama `/api/generate` liest weder `raw` noch `system` und vervollständigt immer roh
- **Zeit:** 2026-10-19 23:45 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OllamaApiSupport.java` (`isRaw`, `toOpenAiChatRequestFromGenerate`, `Generated.fromChat`), `OpenAiCompatServer.java` (`handleOllamaGenerate`)
- **Schwere:** niedrig (still: ein Instruct-Modell setzt den Text fort statt zu antworten, `system` und Bilder gingen verloren)
- **Historie:** Offener Verdacht vom 2026-10-18 (TODO (16)). Die Route war seit ihrer Einführung „prompt completion / FIM“. Ollama wendet ohne `"raw": true` die Chat-Vorlage des Modells an, mit `system` als Systemnachricht; Continue (`Ollama.ts`) setzt `raw` nur, wenn der Aufrufer es verlangt.
- **Beschreibung:** Jede Anfrage ohne `suffix` ging als roher Prompt an `/v1/completions`; `system`, `images`, `format` und `think` wurden ignoriert.
- **Fix:** Entscheidung des Nutzers: Ollamas Semantik. Mit `suffix` wie bisher Infill, mit `raw: true` wie bisher die rohe Completion (`system` ignoriert — **korrigiert in F215:** Ollama ignoriert es nicht, sondern lehnt `raw` mit `system`, `template` oder `context` mit 400 ab; so jetzt auch hier), sonst der Chat-Weg: aus der Anfrage wird eine `/api/chat`-Anfrage (`system` + Nutzer-`prompt` mit `images`, dazu `options`, `format`, `think`) und über die vorhandene Übersetzung geschickt; die Antwort trägt `response`, `thinking` (aus `reasoning_content`), `done_reason` und die Zählungen. README und `package-info` beschreiben die drei Wege; der modellgestützte `OpenAiServerCompletionIntegrationTest` bekam eine `raw`-Variante.
- **Tests:** `OpenAiCompatServerHttpTest#ollamaGenerateWithoutRawAppliesTheChatTemplateLikeOllama` (Aufzeichnungs-Backend: Chat-Weg, `system`, Bild, `thinking`, Zählung), `#ollamaGenerateWithRawCompletesTheBarePromptAndIgnoresSystem`; `OpenAiServerCompletionIntegrationTest#ollamaGenerateRawRoundTrip` (Modell, CI).
- **Rot-Grün-Nachweis:** ohne den Fix rot (`was "completions"`), mit ihm grün (96 Tests); der F178-Test gab nun explizit `raw: true` an, weil er die Completion-Abbildung prüft; SpotBugs grün. Die modellgestützte Variante läuft nur in CI bzw. beim lokalen Agenten.
- **Nachtrag (Branch-Review 2026-10-20):** Die Weitergabe von `model`, `options`, `format` und `think` in die synthetische Chat-Anfrage hatte keinen Test; jedes der vier Felder ließ sich aus `GENERATE_CHAT_FIELDS` streichen, ohne dass etwas rot wurde (Fund J3-5). `OllamaApiSupportTest#generateWithoutRawCarriesOptionsFormatAndThinkIntoTheChatRequest` pinnt `model`, `temperature` 0.2, `max_tokens` 16, `stop`, `response_format` `json_object`, `reasoning_effort` `none` und die Reihenfolge System/User; `#generateWithASchemaFormatAndNoSystemIsOneUserMessageWithAJsonSchema` den Schema-Zweig von `format` und die eine User-Nachricht ohne `system`. Gegenprobe je gestrichenem Feld rot: `model` (`Expected: is "m" but: was ""`), `options` (`Expected: is <0.2> but: was <0.0>`), `format` (`Expected: is "json_object" but: was ""`, dazu der Schema-Test), `think` (`Expected: is "none" but: was ""`); unverändert 44/44 in der Klasse grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `f12d970`

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
- **Nachtrag (Branch-Review 2026-10-20):** Die Beschreibung stimmte für die Responses-Übersetzung nicht: deren Feldliste `CHAT_PASSTHROUGH_FIELDS` filtert, was den Mapper erreicht, und enthielt `logit_bias`/`n_probs` nicht; nachgezogen mit F226.
- **Status:** gefixt (der Rest unter (14), Wahrscheinlichkeiten im gestreamten `/v1/completions`, mit F186)
- **Commit:** java-llama.cpp `cceaaa0`

<a id="f181"></a>
### F181 – java-llama.cpp: JNI-Schalen geben Fehlermeldungen als Modified UTF-8 zurück
- **Zeit:** 2026-10-20 01:10 UTC
- **Datei:** `llama/src/main/cpp/jni_utf8.hpp` (neu), `jllama.cpp` (`throw_api_status`, Modell-Laden), `jni_helpers.hpp` (`report_cxx_exception_impl`), `native_server.cpp`, `rpc_bridge.cpp`, `train_engine.cpp`
- **Schwere:** niedrig bis mittel (still verfälschte Meldung auf HotSpot, Prozessabbruch unter Android-CheckJNI)
- **Historie:** Offener Verdacht vom 2026-10-18 (Musterdurchsicht nach F71, TODO (6)). Gemessen auf HotSpot 21 mit einer eigenen JNI-Sonde: `GetStringUTFChars` liefert für U+1F600 `ED A0 BD ED B8 80` (CESU-8), `NewStringUTF` macht aus `F0 9F 98 80` ein falsches Zeichen und verschluckt Folgebytes.
- **Beschreibung:** `ThrowNew` und `NewStringUTF` lesen Modified UTF-8. Die Meldungen aus llama.cpp zitieren Pfade und Anfragetext in Standard-UTF-8. Die Trainer-Meldung für `missing-😀.gguf` kam als `missing-`, U+00F0 U+009F U+0098 U+0080 und `.g` an (der Rest des Pfads fehlte). Die Eingabeseite des Trainers (`GetStringUTFChars`) ist dagegen nicht erreichbar: `TrainingParameters.escapeNonAscii` (`ca6ccdf`) schreibt jedes Nicht-ASCII-Zeichen als `\u`-Escape (Jacksons `ESCAPE_NON_ASCII` ist standardmäßig aus), die Konfiguration ist also dort ASCII, wo sich die Kodierungen unterscheiden (gemessen: `"Hallo \\ud83d\\ude00 Welt"`). Die Fehlermeldungen von Anfragen an das Modell zitieren in den gemessenen Fällen keinen Anfragetext (Probe mit einem Zufallsmodell).
- **Fix:** Neuer Header `jni_utf8.hpp` (nur `jni.h`, keine gecachten IDs, deshalb auch in `train_engine.cpp` nutzbar): `throw_utf8_impl` und `new_string_utf8_impl`. ASCII bleibt bei `ThrowNew`/`NewStringUTF`, alles andere geht als Bytes durch `String(byte[], "UTF-8")`, und Javas Dekoder ersetzt ungültige Folgen durch U+FFFD. Umgestellt sind alle Stellen mit fremdem Text: `throw_api_status`, das Modell-Laden, `report_cxx_exception_impl` (die JNI-Ausnahmegrenze), `NativeServer`, `RpcServer` und der Rückgabetext von `LlamaTrainer`. Stellen mit literalem ASCII-Text bleiben, wie sie sind. Die Eingabeseite des Trainers bekommt einen Kommentar, warum sie sicher ist.
- **Tests:** C++ `JniUtf8.*` (6 in `test_jni_helpers.cpp`: ASCII bleibt bei `ThrowNew`, 4-Byte-Zeichen und ungültiges Byte erreichen den Konstruktor als Originalbytes, dasselbe für `new_string_utf8_impl` und für `jni_guard_impl`); Java `LlamaTrainerTest#aFourByteCharacterInAPathComesBackIntactInTheErrorMessage` (modellfrei, braucht nur `libjllama`; überspringt, wo die Dateinamen-Kodierung den Pfad nicht darstellen kann) und `#aFourByteCharacterInTheTrainingTextReachesTheTrainer` (hält die Maskierung durch `escapeNonAscii` fest, von der die Eingabeseite abhängt).
- **Rot-Grün-Nachweis:** C++: mit durchreichenden Platzhaltern 4 von 6 rot, mit dem Fix 6 von 6 grün, die ganze Suite 640/640. Java gegen die alte Bibliothek rot (`but: was "…/missing-ð.g"`), gegen die neu gebaute grün; `LlamaLoggerTest`, `NativeLibraryLoadSmokeTest`, `RpcServerTest` grün.
- **Nachtrag (Branch-Review 2026-10-20):** Der Satz „Die Fehlermeldungen von Anfragen an das Modell zitieren in den gemessenen Fällen keinen Anfragetext“ gilt für b11639 nicht: upstream zitiert ihn z. B. in `server-common.cpp` (`response_format type must be one of … but got: <type>`), `common/chat.cpp` (`Missing 'role' in message: <message>`) und im Schema-Konverter (`cannot resolve $ref <ref>`), alle drei über `throw_api_status`. Diesen Weg hatte kein Java-Test; ein Rückbau auf `ThrowNew` blieb grün (Fund J11-1). Neu, modellfrei: `NativeLibraryLoadSmokeTest#anErrorMessageKeepsAFourByteCharacterOfTheRequest` (`LlamaModel.jsonSchemaToGrammar("{\"$ref\":\"#/$defs/😀\"}")`, die `LlamaException` muss `#/$defs/😀` enthalten). Gegenprobe mit `env->ThrowNew` in `throw_api_status`: rot (`got: JSON schema conversion failed: JSON schema error at #: cannot resolve $ref #/$defs/ð\x9f\x98\x80, $defs not fo`, der Rest der Meldung fehlt); mit `throw_utf8_impl` 7/7 in der Klasse grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `50b45f8`
- **Nachtrag (Branch-Review 2026-10-20, Funde J9-5/J11-2):** Beschreibung und Tests schrieben die Maskierung der Eingabeseite Jackson zu; sie kommt von `TrainingParameters.escapeNonAscii` (`ca6ccdf`), die gemessene Ausgabe mit kleingeschriebenem Hex (`\ud83d`) ist deren `%04x`. Jackson maskiert nur U+0000 (als Steuerzeichen) selbst. Beide Zeilen oben sind korrigiert, ebenso der Kommentar in `train_engine.cpp` und die Javadoc von `LlamaTrainerTest#aFourByteCharacterInTheTrainingTextReachesTheTrainer`, die jetzt `TrainingParametersTest#nonAsciiTextIsEscapedSoTheJsonCrossesJniIntact` als Unit-Wächter nennt. Nur Doku/Kommentar.

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
- **Nachtrag (Branch-Review 2026-10-20):** `#aRejectionAfterAHeartbeatIsAnEvent` hing an der Uhr: das Backend schlief 300 ms und warf, das 200 kam nur, wenn der 50-ms-Heartbeat in dieser Zeit lief (Fund J1-5). Das Backend wartet jetzt auf einen Latch, den der Test erst zählt, wenn er `: ping` aus dem zeilenweise gelesenen Stream gelesen hat; danach `startsWith(": ping")`, `invalid_request_error` und der Ablehnungsgrund im Text. Dasselbe für das ältere `#streamingEmitsHeartbeatsDuringAGap` (`SlowFakeBackend`, gleiche 50-gegen-300-ms-Form, vorher nur `containsString(":")`). Gegenprobe: mit einem Heartbeat, der nicht schreibt, sind beide deterministisch rot (`Expected: is <200> but: was <500>`, das Backend gibt nach 30 s auf); unverändert 69/69 grün.
- **Nachtrag (Branch-Review 2026-10-20):** Die beim Fix umgeschriebene Fehlerzeile der Agent-README (`llama-atmosphere-agent/README.md`, „error before the stream starts (401, 413, 500, …)“ | *wire*) nannte für 413 eine Abdeckung, die kein Agent-Test hatte; die 413 prüfte nur `OpenAiCompatServerHttpTest#oversizedRequestBodyRejectedWith413` im Kern (Fund J15-1). Statt die Zeile zu kürzen, macht `AtmosphereWireContractTest#oversizedRequestIsRejectedBeforeReachingTheBackend` sie wahr: ein Server mit `maxRequestBodyBytes(64)` (der Systemprompt allein ist länger), der Zug endet mit einem Fehler, der „Request body too large“ enthält, ohne Text, und das Backend sieht keine Anfrage. Gegenprobe gegen einen eigens installierten Kern: beide Grenzen aus (`getMaxRequestBodyBytes()` liefert `Integer.MAX_VALUE`) rot (`Expected: is "" but: was "never"`); nur die `Content-Length`-Vorprüfung aus bleibt grün, weil das begrenzte Lesen ebenfalls 413 antwortet – der Test pinnt den Vertrag, nicht eine der beiden Grenzen. Ob Atmosphere eine 4xx nicht wiederholt, sieht der Test nicht (`RetryPolicy.NONE`). Unverändert 8/8 in der Klasse grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `9587087`

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
- **Commit:** java-llama.cpp `2b62481`
- **Nachtrag (Branch-Review 2026-10-20, Fund J8-3):** Die Javadoc von `InferenceParameters#withVerbose` sagte, `__verbose` stehe am letzten Chunk eines Streams. Upstream (`server-task.cpp`, b11639) setzt es im Chat-Stream an den ersten Chunk des Endergebnisses (`deltas.front()`: ein verbliebenes Tool-Call-Delta, sonst der Chunk mit `finish_reason`), nie an den Usage-Chunk, den `stream_options.include_usage` anhängt und den `OpenAiCompatServer` auf den Übersetzer-Routen immer verlangt; im Text-Completion-Stream an jeden Chunk; die native Antwort trägt es nicht, sie ist das Objekt selbst. Die Javadoc sagt das jetzt, die Klassen-Javadoc von `ChatStreamChunkParser` nennt die Reihenfolge des Endergebnisses (Tool-Call-Deltas, `finish_reason`, Usage). Nur Doku; `AnthropicStreamTranslatorTest#theMatchedStopWordIsTheStopSequenceOfTheMessageDelta` pinnt das Lesen an einem nicht letzten Chunk.

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
- **Commit:** java-llama.cpp `ffb3a8c`

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
- **Commit:** java-llama.cpp `a8faaef`

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
- **Nachtrag (Branch-Review 2026-10-20):** Der Abbruchweg hatte keinen Test: die IOException der Sink kreuzt `streamCompletionsOai` als `UncheckedIOException`, dessen `finally` den nativen Task abbricht, und `LlamaModelBackend` packt sie wieder aus, damit der Server den Disconnect-Zweig (DEBUG) nimmt und nicht den Fehlerzweig (WARN plus Fehler-Event an den gegangenen Client) (Fund J2-7). Das doppelte Weiterreichen aus `stream` und `streamCompletions` ist jetzt der paketprivate Helfer `LlamaModelBackend.relay`, modellfrei getestet in `LlamaModelBackendTest` (die IOException der Sink kommt als dieselbe Instanz zurück und beendet den Modellaufruf, eine fremde `UncheckedIOException` und jede andere RuntimeException laufen unverändert durch). Modellgestützt in `OpenAiServerCompletionIntegrationTest`: `#aSinkThatFailsMidStreamCancelsTheNativeTask` (eine Sink, die beim ersten Chunk wirft; dieselbe Exception-Instanz, dann Warten bis kein Slot mehr arbeitet, 30 s) und `#aClientThatDisconnectsMidStreamIsADisconnectAndFreesTheSlot` (roher Socket, nach der ersten `data:`-Zeile geschlossen; LogCaptor: genau ein DEBUG „client disconnected during stream“, kein WARN „streaming completion failed“). Gezählt wird über `getDecodeTotal()` bei festem Budget (100 Token, `ignore_eos`): das Warten auf einen freien Slot allein ist auch ohne den Abbruch grün, der Slot erzeugt dann nur erst das ganze Budget. Das Budget muss in den Kontext eines Slots passen; mit 200 Token hielt der Slot des Winzmodells (128) von selbst an, und ein fehlender Abbruch war nicht zu unterscheiden. Gegenprobe mit dem Winzmodell: ohne den `finally`-Abbruch in `streamCompletionsOai` sind beide rot (`decodes of the cancelled stream: 100` / `decodes of the disconnected stream: 100`, `Expected: is <true> but: was <false>`); ohne das Auspacken in `relay` sind `LlamaModelBackendTest#anIOExceptionOfTheSinkComesBackAsTheSameInstance` (`expected: <java.io.IOException> but was: <java.io.UncheckedIOException>`) und der Socket-Test rot (`[] Expected: is <1> but: was <0>`); unverändert grün in vier Wiederholungen. Die LEST-Unterdrückung in `spotbugs-exclude.xml` nennt jetzt `relay`.
- **Nachtrag (Branch-Review 2026-10-20):** Das Zählkriterium des vorigen Nachtrags war ein Wettlauf (Nachprüfung zu Fund J2-7, 2026-10-21): der native Worker dekodiert ohne Gegendruck weiter, während die Java-Seite den Abbruch erst nach dem ersten Chunk absetzt; mit intaktem Abbruch maß die Prüfung 2 bis 68 bzw. 4 bis 85 Dekodierschritte von 100, und einer von fünf Läufen war rot (`decodes of the disconnected stream: 85`). Jetzt endet der Stream nie von selbst: `ModelFlag.CONTEXT_SHIFT` (`--context-shift`, vom Server-Parser angenommen, `test_model_flags.cpp` grün) und `ModelParameters.enableContextShift()` sind neu (upstream schaltet die Kontextverschiebung bei b11639 standardmäßig ab, `common.h`: `ctx_shift = false`; die Javadoc von `disableContextShift()` behauptete das Gegenteil und ist berichtigt; jede der beiden Methoden entfernt jetzt die Gegenflagge, `ModelParametersExtendedTest#testTheLaterContextShiftCallReplacesTheEarlierOne`, rot ohne das Entfernen: `Expected: not map containing ["--context-shift"->ANYTHING]`). Das Testmodell der Klasse verschiebt seinen Kontext, beide Streams haben kein Token-Limit und `ignore_eos`, und das Kriterium ist allein das Warten auf einen freien Slot (30 s); `getDecodeTotal()` und das Budget sind entfallen. Gegenprobe mit dem Winzmodell: ohne den `finally`-Abbruch in `streamCompletionsOai` sind beide rot (`a slot is still processing after 30000 ms`, der Socket-Test auch allein, im abschließenden Warten), ohne das Auspacken in `relay` ist der Socket-Test rot (`[]`); mit beidem grün in zwölf Wiederholungen.
- **Status:** gefixt
- **Commit:** java-llama.cpp `4fb82c1`

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
- **Commit:** java-llama.cpp `5fff6b8`

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
- **Commit:** java-llama.cpp `5fff6b8`
- **Nachtrag (Branch-Review 2026-10-20):** Der `isEmpty()`-Zweig in `appendCsvValue` ließ einen leeren Vorwert still verschwinden; behoben als F234 (ein leerer Wert wird abgewiesen).

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
- **Commit:** java-llama.cpp `5fff6b8`

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
- **Nachtrag (Branch-Review 2026-10-20):** `#everyControlCharacterSurvivesAsJson` prüfte für die 32 Steuerzeichen nur, dass jeder Chunk parst; jedes `\uXXXX` und jede kurze Maske parst aber, so dass ein falsches Zeichen durchging – auf Linux ungeprüft blieben die Werte von `\r`, `\b` und `\f` (Fund J15-2). Der neue Helfer `contentOf(List<String>)` hängt das `delta.content` aller Chunks eines Zuges aneinander; der Test vergleicht damit jetzt für jedes Zeichen von U+0000 bis U+001F sowie für Anführungszeichen/Backslash und DEL/U+0085/U+2028 den Text mit der Eingabe, und `#theTextComesBackOutOfTheChunkUnchanged` nutzt denselben Helfer statt der eigenen Schleife. Gegenprobe rot: `\r` als `\n` geschrieben (`U+000D Expected: is "a\rb" but: was "a\nb"`; die alte Testfassung blieb damit auf Linux grün, 3/3), `\b` als `\f` (`U+0008`); unverändert 3/3 grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `d68aa3d`

<a id="f197"></a>
### F197 – java-llama.cpp: `n` > 1 wird auf jedem nativen Completion-Weg still auf eine Antwort reduziert (das gestreamte `/v1/completions` verlor dabei den Schutz aus F180)
- **Zeit:** 2026-10-20 10:30 UTC
- **Datei:** `llama/src/main/cpp/jllama_engine.cpp` (`populate_completion_task`), `llama/src/test/cpp/test_c_api.cpp`, `llama/src/test/java/net/ladenthin/llama/server/OpenAiServerCompletionIntegrationTest.java`
- **Schwere:** mittel (zwei Antworten verlangt, eine geliefert, ohne Hinweis — dieselbe Klasse wie F184)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20. F180 hatte `n > 1` im Java-Mapper für den gestreamten `/v1/completions` abgelehnt; F186 (`4fb82c1`) verlegte diesen Weg auf den nativen OAI-Pfad und entfernte `toCompletionParameters` und damit die Ablehnung samt ihrer Testzusicherung.
- **Beschreibung:** Upstreams Schema liest `n` als `n_cmpl` mit den harten Grenzen 1 … `n_parallel` (`server-schema.cpp:62`) und beantwortet die weiteren Completions in Kind-Tasks, die der Server anlegt (`server-context.cpp:4888`). Die Engine stellt genau einen Task ein und legt keine Kinder an. Jedes `n` zwischen 2 und der Slot-Zahl wurde also angenommen und mit einer Wahl beantwortet — gemessen vom Reviewer mit dem Zufallsmodell und `setParallel(2)`: `handleCompletionsOai` mit `n:2` eine Wahl, `streamCompletionsOai` ebenso; erst `n:3` scheiterte am Schema. Betroffen: C-API, FFM, `LlamaModel.handle*`, das blockierende `/v1/completions` (leitet den Körper wörtlich weiter) und seit F186 das gestreamte.
- **Fix:** `populate_completion_task` lehnt `task.params.n_cmpl > 1` nach dem Schema mit `jllama_invalid_request` ab ("a request takes one completion, this one asks for 'n' = N; send separate requests"), wie mehrere Prompts (F184). Das deckt die blockierenden, gestreamten und Chat-Wege an einer Stelle.
- **Tests:** `CApi.SeveralCompletionsInOneRequestAreRefusedNotDropped` (eigenes Modell mit `-np 2`; `jllama_completions_oai`, `jllama_completions` mit `n_cmpl`, der Stream-Request und `jllama_chat_completions` werden abgelehnt, `n:1` liefert eine Wahl), `OpenAiServerCompletionIntegrationTest#aCompletionAskingForSeveralChoicesIsAnswered400OnBothRoutes` (beide `/v1/completions`-Routen, das Testmodell jetzt mit `setParallel(2)`), `#severalChoicesAreRefusedOnTheModelsOwnCompletionAndChatPaths`.
- **Rot-Grün-Nachweis:** Der C-API-Test gegen den alten Code rot (alle vier Wege `OK` statt `INVALID_REQUEST`), mit dem Fix grün; `ctest` grün. Die beiden Java-Tests lokal mit dem Zufallsmodell grün (die zwei Infill-Tests der Klasse scheitern dort erwartungsgemäß, das Zufallsmodell hat keine FIM-Tokens); in CI mit dem echten Modell.
- **Status:** gefixt
- **Commit:** java-llama.cpp `7a51bbc`

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
- **Commit:** java-llama.cpp `c07aad0`
- **Nachtrag (Branch-Review 2026-10-20, Fund J13-1):** `llama-ffm/README.md` nannte noch „7 model-free tests“ (Stand vor dem Branch) und behauptete, die nativen Tests übersprängen sich ohne Bibliothek; der CLAUDE.md-Punkt „Tests“ des Moduls kannte nur `JllamaNativeTest` und `FfmLlamaModelIntegrationTest`. Das README nennt jetzt keine Zahl mehr und sagt, welche Klassen ohne Bibliothek laufen (`FfmNativeServerTest`, `FfmTextToSpeechTest`, über Fake-`Backend`s, F59/F198), welche sich überspringen (`JllamaNativeTest`, der modellgestützte Test) und dass `FfmRpcServerTest` eine braucht (er überspringt sich nicht); CLAUDE.md zählt die beiden Lebenszyklus-Klassen, die drei `takeStrings`-Fälle, `FfmRpcServerTest`, `oaiCompletionStreamEndsWithUsage` und die beiden Callback-Fehlerfälle auf. Nur Doku.

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
- **Commit:** java-llama.cpp `aa58171`

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
- **Nachtrag (Branch-Review 2026-10-20):** `runEachOnItsOwnThread_runsNoMoreRequestsAtOnceThanTheBound` behauptete „genau 3 gleichzeitig“ aus 20-ms-Sleeps; das „genau“ hing an der Planung der Threads (Fund J9-2). Jetzt mit zwei Latches statt Sleep: jede Anfrage wartet, bis drei drin sind (das Maximum erreicht die Grenze ohne Zeitbezug), dann bis zu 200 ms auf eine vierte, die nur ein Pool über der Grenze starten kann. Gegenprobe ohne die Grenze (`newFixedThreadPool(requests.size())`) deterministisch rot (`Expected: is <3> but: was <4>`); mit ihr 18/18 in der Klasse grün, dreimal wiederholt.
- **Status:** gefixt
- **Commit:** java-llama.cpp `a2bfad8`
- **Nachtrag (Branch-Review 2026-10-20, Fund J9-6):** Die Begründung der `HES_LOCAL_EXECUTOR_SERVICE`-Ausnahme in `llama/spotbugs-exclude.xml` beschrieb noch einen Thread pro Anfrage, alle zugleich; sie nennt jetzt `min(Batchgröße, maxThreads)` mit `MAX_CHAT_THREADS` = 64 für `chatBatch` und das Herunterfahren im `finally`. Abschnitt 2.4 von `docs/feature-investigation-llama-stack-client-kotlin.md` sprach noch von den asynchronen Wrappern; er nennt jetzt beide Mechanismen (`completeBatch`/`completeBatchWithStats` reichen auf dem aufrufenden Thread ein und empfangen in Eingabereihenfolge, F167; `chatBatch` über den Pool mit höchstens 64 Threads, F200). Die Ausnahme selbst bleibt. Nur Doku.

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
- **Commit:** java-llama.cpp `8e2642f`

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
- **Commit:** java-llama.cpp `29ff8c1`
- **Nachtrag (Branch-Review 2026-10-20, Fund J8-1, Testteil):** upstreams Projektion (`json_get_nested_values`, `server-common.cpp`, b11639) trennt verschachtelte Pfade mit `/`, nicht mit `.`; der Schlüssel der Projektion ist der ganze Pfad, ein unbekannter Pfad fällt ohne Fehler weg. `OpenAiCompatServerIntegrationTest#theDebugObjectTheAnthropicRouteAsksForCarriesOnlyTheStopWord` fragt nur den Schlüssel `stopping_word` ab und sieht den Unterschied nicht. `OpenAiCompatServerIntegrationTest#aNestedResponseFieldIsNamedWithASlashAndKeepsThePathAsItsKey` schickt `verbose: true` mit `response_fields: ["timings/predicted_n"]` (in `__verbose` genau dieser Schlüssel, ganzzahlig > 0) und mit `["timings.predicted_n"]` (`__verbose` leer). Mit der Punkt-Schreibweise in der ersten Anfrage rot („Expected: is <[timings/predicted_n]> but: was <[]>“); lokal mit dem Winzmodell grün (14/14 der Klasse). Das Beispiel in `InferenceParametersTest#withResponseFieldsWritesTheProjectionAsAStringArray` heißt jetzt `timings/predicted_n`; die Javadoc von `withResponseFields` und die CHANGELOG-Zeile folgen in der Doku-Phase.
- **Nachtrag (Branch-Review 2026-10-20, Fund J8-1, Doku-Teil):** Die Javadoc von `InferenceParameters#withResponseFields` nennt jetzt Pfade mit `/` (`timings/predicted_n`, wie `json_get_nested_values` sie trennt), den ganzen Pfad als Schlüssel der Projektion (`{"timings/predicted_n": 12}`, kein verschachteltes `timings`) und dass ein Pfad, den das Ergebnis nicht hat -- ein gepunktetes `timings.predicted_n` ist so einer --, ohne Fehler wegfällt; die CHANGELOG-Zeile sagt dasselbe. Nur Doku.

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
- **Commit:** java-llama.cpp `af84cb5`

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
- **Commit:** java-llama.cpp `da6fa8e`

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
- **Commit:** java-llama.cpp `200cc90`
- **Nachtrag (Branch-Review 2026-10-20, Funde J1-4, J3-3, J16-2):** Beide Regeln sind bewusste Abweichungen von Ollama, nicht Ollamas Verhalten: Ollamas `GenerateHandler` prüft nur `req.Prompt == ""` (der `suffix` fügt dort nur `CapabilityInsert` hinzu), beantwortet also einen leeren Prompt mit `suffix` mit der Ladeantwort, und antwortet bei `keep_alive: 0` ohne Prompt mit `done_reason "unload"`. Hier bleibt der leere Prompt mit `suffix` FIM am Dateianfang (auch bei `"suffix":""`, `hasSuffix` fragt nur nach einem String), und `keep_alive: 0` bekommt die Ladeantwort, weil nichts entladen wird — ein „unload“ wäre eine Meldung über eine Aktion, die nicht stattfand. Die Javadoc von `isLoadOnly`/`toOllamaLoadResponse`, die README-Zeile der Ollama-Routen und `TODO.md` sagen das jetzt; vorher schrieb die Javadoc die `suffix`-Bedingung Ollama zu. Reine Doku, kein Test.

### F213 – java-llama.cpp: Ollama `/api/chat` mit einem `messages`-Element, das kein Objekt ist, schließt die Verbindung ohne Antwort
- **Zeit:** 2026-10-20 19:55 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OllamaApiSupport.java` (`toOpenAiChatRequest`), `llama/src/main/java/net/ladenthin/llama/server/OpenAiCompatServer.java` (`handleOllamaChat`)
- **Schwere:** niedrig (eine Anfrage mit falscher Form bekommt statt 400 eine geschlossene Verbindung; sichtbar nur auf TRACE)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20 (Fund J1-1). Dieselbe Klasse wie F89 (`handleResponses` rief den Übersetzer außerhalb der Fehlerbehandlung auf); `/api/chat` war die letzte Ollama-Route mit diesem Aufbau.
- **Beschreibung:** `handleOllamaChat` rief `OllamaApiSupport.toOpenAiChatRequest` vor dem `try` auf, das Ablehnungen auf 400 und Fehler auf 500 abbildet. Der Übersetzer kopiert jedes Element von `messages` mit `ObjectNode message = ollamaMessage.deepCopy()`; für einen String, `null` oder ein Array wirft der vom Compiler eingefügte Cast eine `ClassCastException`. Sie verließ den Handler über `finally { exchange.close(); }` ohne gesendete Kopfzeilen, der JDK-Server schloss die Verbindung ohne Statuszeile und protokollierte nur auf TRACE — gestreamt wie ungestreamt (die Übersetzung liegt vor der Verzweigung). Ein `messages`, das kein Array ist, wurde zudem still als „keine Nachrichten“ gelesen. upstream (`common/chat.cpp`, b11639) lehnt beides mit `std::invalid_argument` ab, also 400.
- **Fix:** Der Übersetzer prüft die Form und wirft eine `IllegalArgumentException` mit upstreams Wortlaut plus Index (`Expected 'messages[1]' to be an object, got "hi"`, `Expected 'messages' to be an array, got "hi"`); fehlendes und `null` bleiben „keine Nachrichten“. `handleOllamaChat` fasst nur den Übersetzer in ein `try` und beantwortet die `IllegalArgumentException` mit 400 und Ollamas Fehlerkörper, wie `handleResponses` seit F89; eine andere `RuntimeException` bleibt ein protokollierter Serverfehler.
- **Tests:** `OpenAiCompatServerHttpTest.aNonObjectOllamaChatMessageIsA400NotADroppedConnection` (ungestreamt und gestreamt, das Backend wird nicht gerufen), `OllamaApiSupportTest.aNonObjectMessageIsRejectedWithItsIndex` (String, `null`, Array), `OllamaApiSupportTest.messagesThatAreNoArrayAreRejected`.
- **Rot-Grün-Nachweis:** Ohne den Fix rot: der HTTP-Test mit `java.net.SocketException: Unexpected end of file from server` aus `getResponseCode()`, `aNonObjectMessageIsRejectedWithItsIndex` mit „Unexpected exception type thrown, expected: <java.lang.IllegalArgumentException> but was: <java.lang.ClassCastException>“, `messagesThatAreNoArrayAreRejected` mit „Expected java.lang.IllegalArgumentException to be thrown, but nothing was thrown.“; mit dem Fix grün (`OllamaApiSupportTest` 39, `OpenAiCompatServerHttpTest` 65, `OpenAiRequestMapperTest` 35, `LlamaArchitectureTest` 13), `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `6295831`

### F214 – java-llama.cpp: Eine `IOException` des Backends vor dem ersten Chunk gilt als Verbindungsabbruch — keine Antwort, nur DEBUG
- **Zeit:** 2026-10-20 20:09 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiCompatServer.java` (`ResponseStream`; die `IOException`-Zweige von `streamCompletions`, `streamChat`, `streamOllamaChat`, `streamAnthropic`, `streamResponses`)
- **Schwere:** niedrig (nur über die öffentliche Schnittstelle `OpenAiBackend` erreichbar; `LlamaModelBackend` reicht nur die `IOException` der Senke weiter)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20 (Fund J1-2). Eine Folge von F182: vorher sendete jede Streaming-Route die Kopfzeilen vor dem Backend-Aufruf, der Client sah wenigstens ein 200 mit leerem Stream.
- **Beschreibung:** `OpenAiBackend.stream`/`streamCompletions` dürfen laut Javadoc eine `IOException` werfen, wenn „generation fails“; der ungestreamte Pfad beantwortet sie mit 500. Die fünf Streaming-Routen fingen sie mit `LOG.debug("client disconnected during stream", e)` und schrieben nichts. Seit F182 beginnt die Antwort erst mit dem ersten Schreiben, also endete eine solche Anfrage mit `exchange.close()` ohne Kopfzeilen — der JDK-Server schloss die Verbindung ohne Statuszeile, entgegen dem CHANGELOG-Eintrag zu F182 („a failure before it is `400` or `500` with the route's own error body“). Dazu: Ein Client, der gegangen ist, fällt oft erst beim Schreiben der Kopfzeilen auf; `answerIfNotStarted` sah dann `os == null` und hätte die Anfrage ein zweites Mal beantworten wollen.
- **Fix:** Die fünf `IOException`-Zweige beantworten die Anfrage mit 500 und dem Fehlerkörper der Route, solange der Stream nicht begonnen hat (WARN), und bleiben sonst beim DEBUG für den Abbruch. `ResponseStream` merkt sich den Start vor dem Schreiben der Kopfzeilen (`started`), sodass ein dort gescheiterter Start als begonnen gilt: `answerIfNotStarted` gibt `false`, ein weiteres Schreiben wirft, statt die Kopfzeilen erneut zu senden. `answerIfNotStarted` beantwortet eine schon beantwortete Anfrage nicht ein zweites Mal.
- **Tests:** `OpenAiCompatServerHttpTest.aBackendIOExceptionBeforeTheFirstChunkIsA500OnEveryStreamingRoute` (alle fünf Streaming-Routen, `IoFailingBackend`), `ResponseStreamTest.aStreamWhoseHeadersCouldNotBeWrittenHasStarted`.
- **Rot-Grün-Nachweis:** Ohne den Fix rot: der HTTP-Test mit `java.net.SocketException: Unexpected end of file from server` aus `getResponseCode()`, der `ResponseStream`-Test mit „Expected: is <false> but: was <true>“ (`answerIfNotStarted` nach gescheitertem Kopfzeilen-Schreiben); mit dem Fix grün (`ResponseStreamTest` 5, `OpenAiCompatServerHttpTest` 66, `OllamaApiSupportTest` 39, `LlamaArchitectureTest` 13). Modellgestützt mit dem Winzmodell `OpenAiCompatServerIntegrationTest` 13/13 grün; in `OpenAiServerCompletionIntegrationTest` scheitern nur die beiden Infill-Fälle (500, das Winzmodell hat keine FIM-Token), die keine Streaming-Route berühren. `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `d00936b`

### F215 – java-llama.cpp: Ollama `/api/generate` mit `raw: true` verwirft `system`, `template` und `context` still — Ollama antwortet 400
- **Zeit:** 2026-10-20 20:20 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OllamaApiSupport.java` (neu `hasRawConflict`, `RAW_CONFLICT_MESSAGE`; Javadoc von `isRaw`), `llama/src/main/java/net/ladenthin/llama/server/OpenAiCompatServer.java` (`handleOllamaGenerate`)
- **Schwere:** niedrig (der Client bekommt eine plausible Antwort, die seinen Systemprompt nie gesehen hat)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20 (Funde J1-3, J3-2, J16-1 — derselbe Fehler). Eingeführt mit F179, dessen Fix-Zeile, der CHANGELOG-Eintrag und der Test `ollamaGenerateWithRawCompletesTheBarePromptAndIgnoresSystem` das stille Verwerfen als Ollamas Verhalten beschrieben bzw. festschrieben.
- **Beschreibung:** Der `raw`-Zweig von `handleOllamaGenerate` schickte `toOpenAiCompletionRequest` an das Backend, das nur `prompt`, Sampling und `stop` übernimmt; `system`, `template` und `context` gingen ohne Meldung verloren. Ollamas `GenerateHandler` (`server/routes.go`, seit v0.3.0 unverändert) prüft zuerst `if req.Raw && (req.Template != "" || req.System != "" || len(req.Context) > 0)` und antwortet 400 `{"error":"raw mode does not support template, system, or context"}` — vor der Ladeantwort und vor dem Suffix-Weg.
- **Fix:** `OllamaApiSupport.hasRawConflict` prüft genau Ollamas Bedingung (nicht leeres `system`/`template`, nicht leeres `context`-Array; leere Werte sind kein Konflikt); `handleOllamaGenerate` beantwortet einen Konflikt direkt nach dem Einlesen, vor Ladeantwort und FIM, mit 400 und Ollamas Wortlaut, ohne das Backend zu rufen — gestreamt wie ungestreamt. CHANGELOG-Zeile und die Fix-Zeile von F179 sind korrigiert. Nicht Teil dieses Fixes: `format` und `images` im `raw`-Weg (Ollama behält sie, hier fallen sie weg) und `template`/`context` im Weg ohne `raw` (Ollama wendet sie an) — beides offen.
- **Tests:** `OpenAiCompatServerHttpTest.ollamaGenerateWithRawAndASystemPromptIsRejectedLikeOllama` (`system`, `template`, `context`, mit `suffix`, ohne Prompt; das Backend wird nicht gerufen) ersetzt `ollamaGenerateWithRawCompletesTheBarePromptAndIgnoresSystem`; `ollamaGenerateWithRawCompletesTheBarePrompt` (der Weg ohne Konflikt); `OllamaApiSupportTest.rawWithSystemTemplateOrContextConflictsLikeOllama` (vier Konflikte, vier zulässige Formen).
- **Rot-Grün-Nachweis:** Gegen den unveränderten Handler (mit einem `hasRawConflict`, das `false` liefert) rot: der HTTP-Test mit „Expected: is <400> but: was <200>“ für `{"model":"test-model","prompt":"once upon","system":"x","raw":true,"stream":false}`, der Einheitstest mit „Expected: is <true> but: was <false>“; mit dem Fix grün (`OllamaApiSupportTest` 40, `OpenAiCompatServerHttpTest` 67, `LlamaArchitectureTest` 13), `spotbugs:check` und `javadoc:javadoc` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `7933eb0`

### F216 – java-llama.cpp: Ollama `/api/chat` ohne Nachrichten (Ollamas Vorladen) wird mit 400 abgewiesen statt mit der Ladeantwort beantwortet
- **Zeit:** 2026-10-20 20:27 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OllamaApiSupport.java` (neu `isChatLoadOnly`, `toOllamaChatLoadResponse`), `llama/src/main/java/net/ladenthin/llama/server/OpenAiCompatServer.java` (`handleOllamaChat`)
- **Schwere:** niedrig (Clients, die über die Chat-Route vorladen — `ollama-python` `chat(model)` ohne Nachrichten, Open WebUI —, bekommen einen Fehler)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20 (Fund J3-1). Das Gegenstück zu F205, das die Ladeantwort nur für `/api/generate` einführte.
- **Beschreibung:** Ollama beantwortet ein `/api/chat` ohne Nachrichten mit dem Laden des Modells (`ChatHandler`: `if len(req.Messages) == 0` → `message:{role:"assistant",content:""}`, `done: true`, `done_reason "load"`, ein Objekt ohne Blick auf `stream`; in Go ist `len` eines fehlenden oder `null`-Slices 0). Hier übersetzte `handleOllamaChat` die Anfrage zu `messages: []`, und `OpenAiRequestMapper` wies sie mit „'messages' must be a non-empty array“ ab → 400, auch im Streaming-Weg, den ein Body ohne `stream` nimmt.
- **Fix:** `isChatLoadOnly` (`messages` fehlt, ist `null` oder ein leeres Array; ein `messages` falschen Typs ist kein Vorladen und wird von der Übersetzung abgewiesen, F213) und `toOllamaChatLoadResponse` (Ollamas Form); `handleOllamaChat` antwortet damit vor der Übersetzung und ohne das Backend, auch bei `stream` ein einzelnes Objekt. `keep_alive: 0` (bei Ollama das Entladen) bekommt wie bei F205 die Ladeantwort: das Modell dieses Servers bleibt geladen. `/v1/chat/completions` weist leere `messages` weiter ab.
- **Tests:** `OpenAiCompatServerHttpTest.ollamaChatWithoutMessagesOnlyReportsTheModelLoaded` (ohne `messages`, `[]`, `null` mit `stream`, `keep_alive: 0`; das Backend wird nicht gerufen — `RecordingBackend` zeichnet dafür jetzt auch `stream` auf), `OllamaApiSupportTest.aChatRequestWithoutMessagesOnlyLoadsTheModel`.
- **Rot-Grün-Nachweis:** Gegen den unveränderten Handler (mit Stubs, die `false` bzw. `{}` liefern) rot: der HTTP-Test mit „Expected: is <true> but: was <false>“ für `done` bei `{"model":"test-model"}` (das Testbackend streamte eine Antwort; das echte Backend weist die leere Konversation mit 400 ab), der Einheitstest mit „Expected: is <true> but: was <false>“; mit dem Fix grün (`OllamaApiSupportTest` 41, `OpenAiCompatServerHttpTest` 68, `LlamaArchitectureTest` 13), `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `a4d0604`

### F217 – java-llama.cpp: Ollamas `think: true` wird verworfen — ein Server mit abgeschaltetem Denken übergeht es still
- **Zeit:** 2026-10-20 20:31 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OllamaApiSupport.java` (`toOpenAiChatRequest`; über `GENERATE_CHAT_FIELDS` auch `/api/generate`)
- **Schwere:** niedrig (nur bei einem Server, der das Denken per Vorgabe abschaltet; keine Regression — vor F78 wurde `think` ganz ignoriert)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20 (Fund J3-4). F78 übersetzte `think: false` und die Stufen, ließ `true` aber „bei der Vorgabe der Vorlage“; die Fix-Zeile von F78 ist korrigiert.
- **Beschreibung:** upstream (`server-common.cpp`, b11639) setzt `inputs.enable_thinking` zuerst auf die Vorgabe des Servers (`--reasoning off` schaltet sie ab, ebenso das veraltete `--chat-template-kwargs '{"enable_thinking":false}'`) und lässt dann ein `enable_thinking: true` der Anfrage darüber gewinnen. Ein OpenAI-Client kann das Denken also ausdrücklich einschalten; Ollamas `think: true` („should the model think before responding“) wurde dagegen verworfen und von einem solchen Server still übergangen. (`--reasoning-budget 0` ist ein Sampler-Budget, das `enable_thinking` nicht aufhebt; das bleibt unverändert.)
- **Fix:** `think: true` wird zu `chat_template_kwargs: {"enable_thinking": true}` als JSON-Boolean (upstream weist einen String ab); `false` und die Stufen bleiben wie mit F78. Auf einer Vorlage ohne Denken ist der Schalter wie bei upstreams OpenAI-Route eine wirkungslose Template-Variable, kein Fehler (Ollama selbst antwortet dort 400) — bewusst wie die OpenAI-Oberfläche, über die die Übersetzung läuft.
- **Tests:** `OllamaApiSupportTest.thinkTrueSwitchesThinkingOnExplicitly` (Übersetzer → `OpenAiRequestMapper` → `InferenceParameters.toJson()`, für `/api/chat` und `/api/generate`).
- **Rot-Grün-Nachweis:** Ohne den Fix rot: „Expected: is <true> but: was <false>“ (`enable_thinking` fehlte in `chat_template_kwargs`); mit dem Fix grün (`OllamaApiSupportTest` 42, `OpenAiRequestMapperTest` 35, `OpenAiCompatServerHttpTest` 68, `LlamaArchitectureTest` 13), `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `eade620`

### F218 – java-llama.cpp: Chat-Anfrage mit `json_schema` und `grammar` verliert das Schema still statt upstreams 400
- **Zeit:** 2026-10-20 21:24 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiRequestMapper.java` (`applyLlamaCppSampling`)
- **Schwere:** mittel (stiller Fallback: der Client bekommt 200 und eine Antwort ohne sein Schema)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20 (Fund J2-1). F95 führte die Regel „`json_schema` nur ohne `grammar`, wie upstream“ ein; das stimmte für das Completion-Schema, das der Mapper damals auch für das gestreamte `/v1/completions` bediente (`server-schema.cpp`, der Alias-Handler lässt die Grammatik gewinnen). Seit F186 reicht `/v1/completions` den Körper wörtlich durch, der Mapper bedient nur noch die beiden Chat-Wege, und deren Parser entscheidet anders.
- **Beschreibung:** Beide Chat-Wege übergeben das gemappte JSON an upstreams `oaicompat_chat_params_parse` (`jllama_engine.cpp`, `parse_oai_chat_params`). Der weist einen Körper mit beiden Feldern ab: `throw std::runtime_error("Cannot use both json_schema and grammar")` (`server-common.cpp:1300-1304`, b11639) → `jllama_invalid_request` → `InvalidRequestException` → 400, so wie llama-server selbst antwortet. Der Mapper ließ das Schema neben einer Grammatik aber fallen, die Ablehnung konnte nie greifen, und `#jsonSchemaReachesTheMappedRequestUnlessAGrammarIsGiven` pinnte das Verwerfen. Nicht betroffen: `response_format` + `grammar` weist upstream nicht ab (`common/chat.cpp` lässt das Schema gewinnen); `response_format` reicht der Mapper ohnehin wörtlich durch.
- **Fix:** `grammar` und `json_schema` werden unabhängig voneinander übernommen; über das Paar entscheidet upstreams Chat-Parser mit seinem eigenen Text. Keine Ablehnung im Mapper, damit die Regel bei einem llama.cpp-Bump nicht auseinanderläuft. Javadoc von `applyLlamaCppSampling` berichtigt, die Fix-Zeile von F95 ebenso.
- **Tests:** `OpenAiRequestMapperTest#jsonSchemaAndGrammarAreBothForwardedSoUpstreamCanRefuseThePair` (ersetzt die zweite Hälfte von `#jsonSchemaReachesTheMappedRequestUnlessAGrammarIsGiven`, deren erste Hälfte als `#jsonSchemaReachesTheMappedRequest` bleibt); modellgestützt `ErrorHandlingTest#testJsonSchemaWithGrammarInAChatIsAnInvalidRequest` (ein Chat mit beiden Feldern ist eine `InvalidRequestException` mit upstreams Text „Cannot use both json_schema and grammar“ — der Beleg, dass das Durchreichen beim Client als 400 ankommt).
- **Rot-Grün-Nachweis:** Ohne den Fix rot: „Expected: is "object" but: was ""“ (`json_schema` fehlte im gemappten Körper); mit dem Fix grün (`OpenAiRequestMapperTest` 36/36). `ErrorHandlingTest#testJsonSchemaWithGrammarInAChatIsAnInvalidRequest` grün mit dem lokalen Winzmodell (CPU); in der Klasse scheitert unabhängig davon `#testHandleInfillMissingInputSuffix`, weil das Winzmodell keine FIM-Token hat. `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `fea0a1d`

### F219 – java-llama.cpp: Chat-Mapper normalisiert `max_tokens` unter -1 und `n` unter 1 still auf die Vorgaben
- **Zeit:** 2026-10-20 21:30 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiRequestMapper.java` (`applyCommonFields`)
- **Schwere:** niedrig
- **Historie:** Gefunden beim Review des Branches am 2026-10-20 (Fund J2-3). Dieselbe Form wie F102/F103 (der Mapper behandelt einen Wert anders als upstream), nur auf der anderen Seite des Wertebereichs: F103 übernahm die Obergrenze ab `>= 0`, F180 lehnte `n > 1` ab.
- **Beschreibung:** upstreams Schema (`server-schema.cpp`, b11639) deklariert `n_predict` (Aliase `max_completion_tokens`, `max_tokens`) mit den harten Grenzen -1 … INT32_MAX und `n_cmpl` (Alias `n`) mit 1 … `n_parallel`; eine Verletzung ist `std::invalid_argument` → `InvalidRequestException` → 400. Der Mapper setzte `n_predict` nur ab `>= 0`: `"max_tokens": -2` (ebenso `n_predict`/`max_completion_tokens`) lief auf den gemappten Wegen ohne Obergrenze, das wörtlich durchgereichte `/v1/completions` antwortet 400. `"n"` erreicht upstream von hier nie (es ist kein `RequestField`), und der Mapper lehnte nur `> 1` ab — `"n": 0` und `-1` liefen still mit einer Antwort. Betroffen: beide Chat-Wege, damit auch Anthropic `/v1/messages` und Responses `/v1/responses` (sie kopieren die Obergrenze, wie der Client sie schickt); Ollama normalisiert `num_predict < -1` schon vorher auf -1 (F104).
- **Fix:** Jede Obergrenze außer -1 wird durchgereicht, upstreams Schema weist -2 und darunter mit seinem eigenen Text ab; ein ausdrückliches -1 bleibt wie ein fehlendes Feld ungesetzt (upstream nimmt dann `--predict` des Servers, `server-context.cpp`). `n` ungleich 1 ist eine ungültige Anfrage (`IllegalArgumentException` → 400, „'n' must be 1 (one answer is returned) but was N“); die Ablehnung bleibt im Mapper, weil `n` upstream von hier nicht erreicht. Der Kommentar über `0` („generate nothing“) ist aus dem Mapper entfernt, weil er `withNPredict` widersprach; die übrigen Stellen zu `0` gehören zu einem eigenen Fund.
- **Tests:** `OpenAiRequestMapperTest#tokenCapBelowMinusOneReachesUpstreamsRefusal`, `#zeroOrNegativeChoicesAreRefusedLikeUpstream` (`n` 0 mit Meldung, `n` -1), als Schutz `#explicitTokenCapMinusOneLeavesTheServerDefault`; modellgestützt `ErrorHandlingTest#testNPredictMinusTwoInAChatIsAnInvalidRequest` (ein Chat mit `n_predict` -2 ist eine `InvalidRequestException`, die `n_predict` nennt — der Beleg, dass das Durchreichen beim Client als 400 ankommt).
- **Rot-Grün-Nachweis:** Ohne den Fix rot: „Expected: is <-2> but: was <-1>“ (`n_predict` fehlte) und „Expected java.lang.IllegalArgumentException to be thrown, but nothing was thrown.“ (`n: 0`); mit dem Fix grün (`OpenAiRequestMapperTest` 39/39, `OpenAiCompatServerHttpTest` 68/68, `AnthropicApiSupportTest` 26/26, `ResponsesApiSupportTest` 54/54, `OllamaApiSupportTest` 42/42). `ErrorHandlingTest#testNPredictMinusTwoInAChatIsAnInvalidRequest` grün mit dem lokalen Winzmodell (CPU); `#testHandleInfillMissingInputSuffix` scheitert unabhängig davon am Winzmodell ohne FIM-Token. `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `63f05e1`

### F220 – java-llama.cpp: Chat-Mapper verwirft `top_logprobs` ohne `logprobs` still und liest ein ganzzahliges `logprobs` als `n_probs` 20
- **Zeit:** 2026-10-20 21:40 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OpenAiRequestMapper.java` (`applyLogprobs`, neu; vorher in `toInferenceParameters`)
- **Schwere:** niedrig
- **Historie:** Gefunden beim Review des Branches am 2026-10-20 (Fund J2-4). F123 übernahm `logprobs: true` → `n_probs`, ließ upstreams zwei Ablehnungen aber bewusst weg und las `logprobs` mit Jacksons `asBoolean`.
- **Beschreibung:** upstreams `oaicompat_chat_params_parse` (`server-common.cpp:1456-1473`, b11639) liest `logprobs` nur als Boolean (`json_value` liefert für eine Zahl oder einen String den Standard `false`), setzt bei `true` `n_probs = top_logprobs` (20 ohne), weist sonst ein nicht-null `top_logprobs` mit „top_logprobs requires logprobs to be set to true“ ab (400; die OpenAI-Spezifikation verlangt dasselbe) und kopiert den Rest des Körpers durch — das Schema liest `logprobs` dann als Alias von `n_probs` (`server-schema.cpp:179-180`, nur ohne `n_probs`). Der Mapper dagegen (a) verwarf ein `top_logprobs` ohne `logprobs` still, weil das gemappte JSON es nicht mehr trug, und (b) las mit `asBoolean` jede von 0 verschiedene Zahl und den String `"true"` als die Chat-Form: `"logprobs": 2` ergab `n_probs` 20 statt upstreams 2, `"n_probs": 5` daneben wurde überschrieben, und ein String, den upstreams Schema mit 400 abweist, lief durch.
- **Fix:** `applyLogprobs` folgt upstream: `logprobs: true` → `n_probs = top_logprobs` (eine Zahl, sonst 20 wie upstreams `json_value`); ein nicht-null `top_logprobs` ohne `logprobs: true` ist eine ungültige Anfrage mit upstreams Text wörtlich (die WEM-Unterdrückung in `spotbugs-exclude.xml` nennt die Methode deshalb); ohne `n_probs` wird eine Zahl als `n_probs` übernommen, jeder andere nicht-null, nicht-boolesche Wert ist eine ungültige Anfrage („'logprobs' must be a boolean or a number“), wie das Schema ihn abweist. `logprobs: false` lässt `n_probs` weiter ungesetzt (upstreams Alias läse 0, die Vorgabe). Weiter offen, nicht Teil dieses Funds: upstreams zweite Ablehnung („logprobs is not supported with tools + stream“) bleibt unerreichbar, weil der Mapper `logprobs` in `n_probs` übersetzt; die Fix-Zeile von F123 ist berichtigt.
- **Tests:** `OpenAiRequestMapperTest#topLogprobsWithoutLogprobsIsRefusedLikeUpstream` (ohne, mit `false`, mit `null` und neben einer Zahl), `#anIntegerLogprobsMapsToNProbsLikeUpstreamsSchemaAlias`, `#nProbsWinsOverAnIntegerLogprobsLikeUpstream`, `#aLogprobsThatIsNeitherBooleanNorNumberIsRefused` (mit `n_probs` daneben keine Ablehnung, wie upstream); als Schutz `#topLogprobsNullWithoutLogprobsIsTheDefault`, die drei F123-Tests unverändert grün.
- **Rot-Grün-Nachweis:** Ohne den Fix vier rot: „Expected java.lang.IllegalArgumentException to be thrown, but nothing was thrown.“ (`top_logprobs: 3` allein bzw. `logprobs: "true"`), „Expected: is <2> but: was <20>“ (`logprobs: 2`), „Expected: is <5> but: was <20>“ (`n_probs: 5` neben `logprobs: 2`); mit dem Fix grün (`OpenAiRequestMapperTest` 44/44, `OpenAiCompatServerHttpTest` 68/68, `AnthropicApiSupportTest` 26/26, `ResponsesApiSupportTest` 54/54, `OllamaApiSupportTest` 42/42, `LlamaArchitectureTest` 13/13). `spotless:apply` grün; `spotbugs:check` meldete zuerst zweimal `WEM_WEAK_EXCEPTION_MESSAGING` für die festen Texte, mit der Unterdrückung grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `ca1baa9`

### F221 – java-llama.cpp: Anthropic meldet ein getroffenes Stoppwort als `stop_reason` `end_turn` statt `stop_sequence`
- **Zeit:** 2026-10-20 22:30 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/AnthropicApiSupport.java` (`toAnthropicResponse`, `anthropicStopReason(String, String)` neu), `AnthropicStreamTranslator.java` (`end`)
- **Schwere:** mittel (ein Client, der auf `stop_reason` verzweigt — die Anthropic-SDKs typisieren es als Aufzählung, Agentenschleifen setzen bei `end_turn` fort —, konnte einen Stopp durch das Wort nicht von einem natürlichen Ende unterscheiden)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20 (Fund J4-1). Folge von F183, das das Wort über `__verbose.stopping_word` erst verfügbar macht und `stop_reason` dabei bewusst wie upstream auf `end_turn` ließ. upstream (`server-task.cpp:723-726`/`:792-795`, b11639 und master) bildet `STOP_TYPE_WORD` wie `STOP_TYPE_EOS` auf `end_turn`/`tool_use` ab und testet den Wert der Spezifikation nirgends.
- **Beschreibung:** Die Messages-API legt fest: „If the model encounters one of the custom sequences, the response stop_reason value will be "stop_sequence" and the response stop_sequence value will contain the matched stop sequence.“ `toAnthropicResponse` und `AnthropicStreamTranslator.end()` bildeten den OpenAI-`finish_reason` unabhängig vom Wort ab, `anthropicStopReason` kannte kein `stop_sequence`: Eine Anfrage mit `stop_sequences: ["X"]`, deren Generierung an `X` endete, bekam das widersprüchliche Paar `stop_reason: "end_turn"`, `stop_sequence: "X"` — blockierend wie gestreamt.
- **Fix:** Neue Überladung `anthropicStopReason(finishReason, stopSequence)`: `stop` mit einem Wort ist `stop_sequence`, alles andere die bisherige Abbildung (`length` → `max_tokens`, `tool_calls` → `tool_use`, sonst `end_turn`). `toAnthropicResponse` liest das Wort jetzt vor dem Stoppgrund, der Stream-Übersetzer übergibt sein gesammeltes Wort. Das Wort ist genau bei einem Wortstopp nicht leer (upstream setzt `stopping_word` nur dort, `server-context.cpp:630`), und ein solcher Stopp hat immer `finish_reason` `stop`, also bleiben `tool_use` und `max_tokens` unberührt. Die Javadoc der Überladung zitiert die Spezifikation und nennt die bewusste Abweichung von upstream. Nicht Teil dieses Commits: `CHANGELOG.md` (dort steht noch „stop_reason stays end_turn“, wird in einem späteren Durchgang nachgezogen); upstream-tauglich als Einzeiler in beiden `to_json_anthropic*`.
- **Tests:** `AnthropicApiSupportTest#aMatchedStopWordIsReportedAsStopReasonStopSequence`, `#theMatchedStopWordIsTheStopSequence` (Erwartung von `end_turn` auf `stop_sequence` umgestellt), `#onlyAStopThatAStopWordEndedIsStopSequence` (die drei unveränderten Zweige mit Wort und `stop` ohne Wort), als Schutz `#withoutAStopWordAStopIsStillEndTurn`; `AnthropicStreamTranslatorTest#aMatchedStopWordIsTheStopReasonStopSequenceOfTheMessageDelta`, `#theMatchedStopWordIsTheStopSequenceOfTheMessageDelta` (umgestellt).
- **Rot-Grün-Nachweis:** Ohne den Fix vier rot: „Expected: is "stop_sequence" but: was "end_turn"“ (beide blockierenden Fälle) und „Expected: a string containing "\"stop_reason\":\"stop_sequence\""“ (beide gestreamten); mit dem Fix grün (`AnthropicApiSupportTest` 29/29, `AnthropicStreamTranslatorTest` 8/8, `OpenAiCompatServerHttpTest` 68/68). `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `7e965d9`

### F222 – java-llama.cpp: Gestreamte Anthropic-Antwort öffnet bei Denkinhalt nach Text einen zweiten Thinking-Block, während der Textblock offen bleibt
- **Zeit:** 2026-10-20 22:35 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/AnthropicStreamTranslator.java` (`onChunk`, `end`, `closeTextBlock` neu)
- **Schwere:** niedrig (braucht einen Parser, der den Denkinhalt nach begonnenem Text verlängert; die SDK-Akkumulatoren sammeln nach Index und setzen den Inhalt trotzdem zusammen, ein sequenziell darstellender Client hängt Text an einen Block, der sichtbar hinter einem späteren Thinking-Block liegt)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20 (Fund J4-3). F81 führte die Thinking-Blöcke im Stream ein, mit dem erklärten Ziel, dass die Blöcke einander folgen, statt wie bei upstream bis zum Ende offen zu bleiben; der Textpfad schloss dafür einen offenen Thinking-Block, der Reasoning-Pfad den Textblock nicht.
- **Beschreibung:** `common_chat_msg_diff::compute_diffs` (`common/chat.cpp:236-252`, b11639) sendet ein `reasoning_content`-Delta, wann immer der geparste Denkinhalt wächst, unabhängig davon, ob schon Text kam, und die PEG-Mapper hängen jeden Reasoning-Knoten an (`chat-peg-parser.cpp:321`, „GPT OSS can have more than 1 reasoning block“; Gemma 4: `message = thought + content`, mehrfach). Für Denken „a“, Text „b“, Denken „c“, Text „d“ lieferte der Übersetzer: Thinking 0 (Start, Delta, Signatur, Stopp), Text 1 (Start, Delta „b“), Thinking 2 (Start, Delta „c“) bei noch offenem Text 1, Signatur und Stopp von 2, dann das Delta „d“ wieder an Index 1 und erst in `end()` den Stopp von 1 — zwei Blöcke gleichzeitig offen, ein Delta an einen Block nach dem Stopp des späteren. Die Streaming-Doku beschreibt „a series of content blocks, each of which has a content_block_start, one or more content_block_delta events, and a content_block_stop event“.
- **Fix:** `closeTextBlock` (Gegenstück zu `closeThinkingBlock`) schließt einen offenen Textblock, bevor ein Thinking-Block geöffnet wird; `end()` nutzt dieselbe Methode. Höchstens ein Block ist offen, das Beispiel ergibt 0 Thinking, 1 Text, 2 Thinking, 3 Text, wie Claude selbst mehrere Thinking-Blöcke streamt. Die Klassen-Javadoc nennt die Regel und die bewusste Asymmetrie: `toAnthropicResponse` kann die Verschränkung nicht abbilden, weil die Chat-Nachricht den Denkinhalt als einen verketteten String trägt.
- **Tests:** `AnthropicStreamTranslatorTest#reasoningAfterTextClosesTheTextBlockBeforeANewThinkingBlock` (zweiter Thinking-Start an Index 2, letztes Textdelta an Index 3, Stopp von 1 vor dem zweiten Thinking-Start, kein Delta an 1 nach dessen Stopp, `end()` schließt genau einen Block, Index 3); die übrigen Tests der Klasse unverändert grün.
- **Rot-Grün-Nachweis:** Ohne den Fix rot: „Expected: a string containing "\"index\":3,\"delta\":{\"type\":\"text_delta\",\"text\":\"d\"}"“ (das Delta „d“ ging an Index 1); mit dem Fix grün (`AnthropicStreamTranslatorTest` 9/9, `AnthropicApiSupportTest` 29/29, `OpenAiCompatServerHttpTest` 68/68). `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `cb123a9`

### F223 – java-llama.cpp: Anthropic `max_tokens`, das keine Zahl ist, wird still durch die Vorgabe 4096 ersetzt statt abgewiesen
- **Zeit:** 2026-10-20 22:40 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/AnthropicApiSupport.java` (`toOpenAiChatRequest`), `OpenAiCompatServer.java` (`handleAnthropicMessages`)
- **Schwere:** niedrig (ein Client mit einem falsch typisierten `max_tokens`, etwa `"100"`, bekam ohne Hinweis eine Obergrenze von 4096 statt seiner)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20 (Fund J4-4). F96 führte die Vorgabe 4096 für ein fehlendes `max_tokens` ein und prüfte dafür „keine Zahl“ statt „fehlt“; vor F96 ließ `copyNumber` einen Nicht-Zahl-Wert ebenso still weg (dann unbegrenzt).
- **Beschreibung:** Die Messages-API typisiert `max_tokens` als Pflicht-Ganzzahl; upstream (`convert_anthropic_to_oai`, `server-chat.cpp:586-589`, b11639) kopiert den Wert wie er ist, und das Schema (`field_num::eval`, `server-schema.cpp:531-549`) weist einen String oder Boolean mit „Field 'max_tokens': … type must be number“ ab (400), behandelt `null` aber wie ein fehlendes Feld (`has_value`, „clients can send null to request the server default“). Die Übersetzung verzweigte auf `isNumber()`: `"max_tokens": "100"` und `true` liefen still mit 4096. Den Wert einfach durchzureichen hilft hier nicht, weil der Mapper (`OpenAiRequestMapper.readMaxTokens`) eine Nicht-Zahl selbst als -1 (unbegrenzt) liest, bevor upstreams Schema sie sieht. Außerdem rief `handleAnthropicMessages` die Übersetzung außerhalb des `try` auf, das eine `IllegalArgumentException` als 400 beantwortet: Eine Ablehnung wäre als abgebrochene Verbindung angekommen.
- **Fix:** Eine Zahl wird kopiert, ein fehlendes Feld oder `null` bekommt wie bisher 4096, jeder andere Wert ist eine `IllegalArgumentException` („'max_tokens' must be an integer but was …“). `handleAnthropicMessages` übersetzt in einem eigenen `try` vor der Verzweigung auf `stream` (wie `handleResponses`) und antwortet mit Anthropics 400 `invalid_request_error`, gestreamt wie blockierend, bevor ein SSE-Strom beginnt. Nicht Teil dieses Funds: Dieselbe stille Klasse besteht auf der OpenAI-Route selbst (`OpenAiRequestMapper.readMaxTokens` liest `"100"` als -1 = unbegrenzt, upstream antwortet 400) — eigener Fund.
- **Tests:** `AnthropicApiSupportTest#aMaxTokensThatIsNotANumberIsRefused` (`"100"` und `true`, Meldung nennt `max_tokens`), als Schutz `#requestWithNullMaxTokensIsCappedLikeAbsent` (vorher und nachher grün) und `#requestWithoutMaxTokensIsCappedLikeUpstream`; `OpenAiCompatServerHttpTest#aStringMaxTokensOnTheAnthropicRouteIsA400InvalidRequestError` (`stream` false und true: 400, `"type":"invalid_request_error"`, nichts generiert).
- **Rot-Grün-Nachweis:** Ohne den Fix rot: „"100" ==> Expected java.lang.IllegalArgumentException to be thrown, but nothing was thrown.“ und „Expected: is <400> but: was <200>“; nur mit der Übersetzungsseite des Fixes scheiterte der HTTP-Test mit „java.net.SocketException: Unexpected end of file from server“ (der Beleg, dass die Ablehnung ohne das verlegte `try` als abgebrochene Verbindung ankäme); mit dem ganzen Fix grün (`AnthropicApiSupportTest` 31/31, `AnthropicStreamTranslatorTest` 9/9, `OpenAiCompatServerHttpTest` 69/69). `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `82bef4a`

### F224 – java-llama.cpp: Responses `input_image` ohne URL und fremde Werkzeugergebnis-Teile werden still verworfen statt abgewiesen
- **Zeit:** 2026-10-20 23:05 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/ResponsesApiSupport.java` (`inputContentWithImages`, `toolOutputParts`, neu `imageUrl`)
- **Schwere:** niedrig (ein Bild per `file_id` oder ein Werkzeugergebnis mit einem `output_text`-Teil kam ohne Hinweis nicht beim Modell an; es antwortete, als wäre es nicht geschickt worden)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20 (Fund J5-1). F17 ließ `file_id`-Bilder bewusst weg („werden weiterhin ignoriert“), F86 übersprang fremde Teiltypen eines Werkzeugergebnisses („upstream weist sie ab“); F89 hatte dieselbe Klasse für `input_file` schon in eine 400 verwandelt.
- **Beschreibung:** `inputContentWithImages` nahm einen `input_image`-Teil nur mit einer nicht leeren URL auf; `{"type":"input_image","file_id":"file-1"}` (nach dem OpenAI-Typ `ResponseInputImageParam` gültig, `image_url` und `file_id` sind beide optional) fiel weg, die Nachricht ging als reiner Text ans Modell. `toolOutputParts` übernahm nur `input_text` und ein `input_image` mit einem String-`image_url`; jeder andere Teil, ein Teil ohne `type` und die Objektform `{"url": …}` (die der Nachrichtenzweig annimmt) fielen weg, ein `[{"type":"output_text",…}]` ergab eine Werkzeugnachricht mit leerem Inhalt. upstream (`convert_responses_to_chatcmpl`, `server-chat.cpp:83-85, 207-216`, b11639) antwortet in jedem dieser Fälle 400: „'image_url' is required“, „Output of tool call missing 'type' field“, „Output of tool call should be 'Input text' or 'Input image'“.
- **Fix:** Ein gemeinsamer Helfer `imageUrl` liest die URL auf beiden Wegen gleich (String oder `{url}`, eine bewusst beibehaltene Erweiterung von F17). Ein `input_image` ohne URL ist eine `IllegalArgumentException` mit upstreams Text plus der `file_id`; im Werkzeugergebnis ebenso ein Teil ohne `type` und ein Teil eines anderen Typs, jeweils mit upstreams Text, dem Typ und der `call_id`. `handleResponses` beantwortet die Ausnahme schon als 400 (F89). Bewusst schärfer als upstream: ein leerer `image_url`-String gilt als fehlend (upstream scheitert daran später in `handle_media`). Nicht geändert: unbekannte Teiltypen einer `message` bleiben Text bzw. werden übergangen, weil derselbe Zweig Assistenten-Nachrichten mit `output_text`/`refusal` trägt (F173).
- **Tests:** `ResponsesApiSupportTest#inputImageWithoutAUrlIsRejectedLikeUpstream`, `#toolOutputPartOfAnotherTypeIsRejectedLikeUpstream`, `#toolOutputPartWithoutATypeIsRejectedLikeUpstream`, `#toolOutputImageWithoutAUrlIsRejectedLikeUpstream`, `#toolOutputImageInObjectFormKeepsItsUrl`; die 400-Abbildung pinnt schon `OpenAiCompatServerHttpTest#responsesWithAnInputFileIsRejectedLikeUpstream`.
- **Rot-Grün-Nachweis:** Ohne den Fix rot (5 Fehlschläge: viermal „Expected java.lang.IllegalArgumentException to be thrown, but nothing was thrown.“, einmal „Expected: is "image_url" but: was ""“), mit dem Fix grün (`ResponsesApiSupportTest` 59/59, `ResponsesStreamTranslatorTest` 5/5, `OpenAiCompatServerHttpTest` 69/69). `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `e372019`

### F225 – java-llama.cpp: Responses `function_call` mit Objekt-`arguments` wird still zu `"{}"`
- **Zeit:** 2026-10-20 23:07 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/ResponsesApiSupport.java` (`appendInput`, Fall `function_call`)
- **Schwere:** niedrig (ein Client, der die Argumente eines früheren Aufrufs als Objekt statt als JSON-Text zurückschickt, bekam einen Verlauf, in dem der Aufruf ohne Argumente gemacht wurde)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20 (Fund J5-2). F172 machte fehlende, leere und nur aus Leerraum bestehende `arguments` zu `"{}"` (Entscheidung des Nutzers, wie F166); vor F172 ergab ein Objekt `""`, das upstreams `json::parse` mit 400 abwies.
- **Beschreibung:** Die Übersetzung las `item.path("arguments").asText("")`. Für ein Objekt, ein Array, eine Zahl oder einen Wahrheitswert liefert Jacksons `asText` bei Containern `""` (Zahlen und Booleans als ihren Text), also wurde `"arguments": {"city":"Paris"}` zu `"{}"` – das Modell sah einen Aufruf, den es so nie gemacht hatte, ohne jeden Hinweis. Der OpenAI-Typ (`ResponseFunctionToolCallParam`) verlangt einen JSON-String, und upstream (`convert_responses_to_chatcmpl`, `server-chat.cpp:163-168, 256-257`, b11639) nimmt ein `function_call`-Item nur mit String-`arguments` an und antwortet sonst 400 („Cannot determine type of 'item'“). F172s Entscheidung deckte nur fehlende/leere Argumente, nicht diesen Fall.
- **Fix:** Ein `arguments`-Wert, der weder fehlt noch `null` noch ein String ist, ist eine `IllegalArgumentException` („'arguments' of a function_call must be a JSON string (got OBJECT, call_id c1)“), die `handleResponses` als 400 beantwortet. Fehlend, `null` und leer bleiben nach F172 `"{}"`, ein String geht unverändert weiter. Die verlustfreie Alternative (das Objekt als JSON-Text weiterreichen) wurde nicht gewählt: sie ginge über den OpenAI-Typ und upstream hinaus.
- **Tests:** `ResponsesApiSupportTest#functionCallWithArgumentsThatAreNotAStringIsRejectedLikeUpstream` (Objekt, Array, Zahl, Boolean; Meldung nennt `'arguments'` und die `call_id`); `#functionCallWithoutArgumentsBecomesAnEmptyArgumentsObject` um `"arguments":null` erweitert (vorher und nachher grün, pinnt, dass `null` weiter `"{}"` ist).
- **Rot-Grün-Nachweis:** Ohne den Fix rot („{"city":"Paris"} ==> Expected java.lang.IllegalArgumentException to be thrown, but nothing was thrown.“), mit dem Fix grün (`ResponsesApiSupportTest` 60/60, `OpenAiCompatServerHttpTest` 69/69). `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `4913acf`

### F226 – java-llama.cpp: Responses-Route verwirft `logit_bias` und `n_probs` still, obwohl der Chat-Mapper sie liest
- **Zeit:** 2026-10-20 23:12 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/ResponsesApiSupport.java` (`CHAT_PASSTHROUGH_FIELDS` und seine Javadoc)
- **Schwere:** niedrig (ein per `logit_bias` gesperrtes Token wurde auf `/v1/responses` trotzdem erzeugt, ohne Hinweis)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20 (Fund J5-3). Die Liste entstand mit F106 (`b370643`) gegen den damaligen Mapper; F180 (`cceaaa0`) brachte dem Mapper `logit_bias` und `n_probs` bei und vermerkte, das erreiche auch die Responses-Übersetzung – die Liste filtert aber, was kopiert wird, und wurde nicht nachgezogen. F183/F202 fügten `verbose` und `response_fields` hinzu, ebenfalls ohne die Liste.
- **Beschreibung:** Upstreams `convert_responses_to_chatcmpl` (`server-chat.cpp:15`, b11639) beginnt mit `json chatcmpl_body = response_body;` und kopiert damit jedes Feld; ein `logit_bias` auf upstreams `/v1/responses` erreicht das Schema (`server-schema.cpp:382`) und wirkt. Hier kopierte die Übersetzung nur die Felder ihrer Liste, deren Javadoc behauptete, es seien die, die der Chat-Mapper liest; `logit_bias` und `n_probs` fehlten, die Anfrage lief ohne sie. Der parametrisierte Test zählte nur die Felder der Liste auf und konnte eine Lücke nicht sehen. Auch die CHANGELOG-Zeile zu F180 (alle Chat-Routen einschließlich Responses reichen `logit_bias`/`n_probs` weiter) stimmte damit nicht.
- **Fix:** `logit_bias` und `n_probs` stehen in der Liste (`n_probs` aus Parität mit upstreams Annahme, sichtbar wird es nicht: die Responses-Ausgabe trägt hier wie upstream keine Wahrscheinlichkeiten). Die Javadoc sagt jetzt, was die Liste ist (die vom Mapper gelesenen Felder, deren Wirkung einen Responses-Client erreichen kann) und nennt die bewussten Auslassungen: die oben übersetzten Strukturfelder, `verbose`/`response_fields` (die Responses-Ausgabe trägt weder das Debug-Objekt noch eine Projektion; weitergereicht zöge `verbose` nur upstreams Debug-Objekt durch JNI, um es zu verwerfen, die Kosten, die F202 auf der Anthropic-Route entfernte) und chats `logprobs`/`top_logprobs` (die Responses-Form ist `top_logprobs` mit `include`, die Ausgabe trägt keine Wahrscheinlichkeiten). Der vorgeschlagene reflektive Wächter über die `request.path(…)`-Literale des Mappers wurde nicht gebaut (er müsste Quelltext lesen und eine große Ausnahmeliste pflegen); die Ausnahmeliste der Javadoc ist der Vertrag.
- **Tests:** `ResponsesApiSupportTest#requestFieldsTheChatMapperReadsPassThroughLikeUpstream` um die Zeilen `logit_bias|{"42":-100}` und `n_probs|3` erweitert.
- **Rot-Grün-Nachweis:** Ohne den Fix rot („Expected: is <{"42":-100}> but: was <>“ und „Expected: is <3> but: was <>“), mit dem Fix grün (`ResponsesApiSupportTest` 62/62, `OpenAiCompatServerHttpTest` 69/69, `OpenAiRequestMapperTest` 44/44). `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `88f823e`

### F227 – java-llama.cpp: `tool_choice`, das eine Funktion benennt, lässt bei mehreren Werkzeugen jedes zu
- **Zeit:** 2026-10-20 23:21 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/ResponsesApiSupport.java` (`toOpenAiChatRequest`, Werkzeugblock), `OpenAiRequestMapper.java` (`toInferenceParameters`, neu `namedToolChoice`)
- **Schwere:** niedrig (mit mehreren Werkzeugen durfte das Modell statt des verlangten ein anderes aufrufen)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20 (Fund J5-4). F109 bildete `{"type":"function","name":…}` (Responses) und `{"type":"function","function":{"name":…}}` (Chat) auf `required` ab und vermerkte, das sei nur „mit einem einzigen Werkzeug genau das verlangte“.
- **Beschreibung:** llama.cpp kennt für `tool_choice` nur `auto`/`none`/`required` (`common_chat_tool_choice_parse_oaicompat`, `common/chat.cpp:314-325`, b11639) und baut die Werkzeug-Grammatik über alle übergebenen Werkzeuge. `required` über die ganze Liste ließ deshalb jeden Aufruf zu; die OpenAI-Semantik („force the model to call a specific function“, `ToolChoiceFunctionParam`) galt nur bei genau einem Werkzeug. Ein unbekannter Name und ein `{"type":"function"}` ohne Namen liefen still als `required` über alle Werkzeuge; OpenAI weist einen unbekannten Namen mit 400 ab. upstream selbst liest das Objekt gar nicht (`json_value(body, "tool_choice", "auto")` fällt auf `auto` zurück), die Übersetzung war upstream also schon voraus.
- **Fix:** Ein benanntes `tool_choice` wird durch Einengen der weitergereichten Werkzeugliste erzwungen: nur das genannte Funktionswerkzeug geht weiter, dazu `required`; so lässt die Grammatik keinen anderen Aufruf zu. Ein Name, den kein Funktionswerkzeug trägt, ist eine `IllegalArgumentException` („tool_choice names unknown tool 'g'“), ein Objekt vom Typ `function` ohne Namen ebenso (beides 400). Beide Wege nutzen denselben Helfer `OpenAiRequestMapper.namedToolChoice`; die Responses-Übersetzung filtert schon beim Kopieren der Werkzeuge. Folge, in der Javadoc genannt: die Chat-Vorlage zeigt dem Modell die anderen Werkzeugdefinitionen nicht mehr (OpenAI lässt sie sichtbar und beschränkt nur die Ausgabe) – mit llama.cpps dreiwertigem `tool_choice` der Preis der Genauigkeit. Die Klassen-Javadoc des Mappers nennt `tool_choice` nur noch in seiner String-Form als unverändert weitergereicht. Nicht Teil dieses Funds: Anthropics `{"type":"tool","name":…}` wird weiter zu `required` über alle Werkzeuge, und ein `tool_choice`-Objekt anderen Typs (`allowed_tools`, `mcp`, …) fällt auf beiden Wegen weiter still auf `auto`.
- **Tests:** `ResponsesApiSupportTest#toolChoiceNamingAFunctionForwardsOnlyThatTool`, `#toolChoiceNamingAnUnknownFunctionIsRejected`, `#toolChoiceOfTypeFunctionWithoutANameIsRejected` und dieselben drei in `OpenAiRequestMapperTest`; die F109-Tests `#toolChoiceNamingAFunctionRequiresAToolCall` (ein Werkzeug) beider Klassen bleiben grün.
- **Rot-Grün-Nachweis:** Ohne den Fix rot (6 Fehlschläge: je Klasse „Expected: is <1> but: was <2>“ und zweimal „Expected java.lang.IllegalArgumentException to be thrown, but nothing was thrown.“), mit dem Fix grün (`ResponsesApiSupportTest` 65/65, `OpenAiRequestMapperTest` 47/47, `OpenAiCompatServerHttpTest` 69/69, `AnthropicApiSupportTest` 31/31, `OllamaApiSupportTest` 42/42, `LlamaArchitectureTest` 13/13). `spotless:apply` und `spotbugs:check` grün (eine erste Fassung mit fester Meldung ohne Namen fiel bei SpotBugs als `WEM_WEAK_EXCEPTION_MESSAGING` durch; die Meldung nennt jetzt das `tool_choice`).
- **Status:** gefixt
- **Commit:** java-llama.cpp `740938d`

### F228 – java-llama.cpp: Responses-Eingabe-Items unbekannten Typs werden zu leeren Nutzernachrichten statt abgewiesen
- **Zeit:** 2026-10-20 23:26 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/ResponsesApiSupport.java` (`appendInput`, `switch` über den Item-Typ; Klassen-Javadoc)
- **Schwere:** niedrig (ein Verlauf mit z. B. Codex' `custom_tool_call`/`custom_tool_call_output` oder einem `item_reference` erreichte das Modell mit leeren Nutzerzügen an Stelle der Aufrufe, ohne Hinweis)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20 (Fund J5-5). Der gemeinsame `case "message": default:`-Zweig stammt aus der Zeit vor dem Branch; der Branch schrieb den `switch` um (Reasoning, Zusammenführung, Werkzeugteile, `input_file`, F17/F86/F89/F99) und kündigt im CHANGELOG an, was nicht bedient werden kann, werde mit 400 abgewiesen.
- **Beschreibung:** `appendInput` las `item.path("type").asText("message")` und führte jeden unbekannten Typ durch den Nachrichtenzweig. Ohne `role` und meist ohne `content` wurde daraus `{"role":"user","content":""}`; die OpenAI-Union `ResponseInputItemParam` kennt über 30 Item-Typen (`custom_tool_call`, `custom_tool_call_output`, `item_reference`, `mcp_call`, `apply_patch_call`, …). upstream (`convert_responses_to_chatcmpl`, `server-chat.cpp:256-257`, b11639) beendet jedes nicht erkannte Item mit „Cannot determine type of 'item'“ (400).
- **Fix:** Der `message`-Fall endet mit `break`; `default` wirft eine `IllegalArgumentException` mit upstreams Text und dem Typ („Cannot determine type of 'item' (custom_tool_call)“), die `handleResponses` als 400 beantwortet. Ein Item ohne `type` (die EasyInputMessage-Form, deren `type` optional ist) landet wie bisher über den Vorgabewert `"message"` im Nachrichtenzweig. Die Klassen-Javadoc nennt die Item-Typen und die Abweisung. Bewusst lockerer als upstream geblieben: eine Nachricht mit einer Rolle außerhalb von user/system/developer/assistant und eine Assistenten-Nachricht ohne `type` werden weiter angenommen (vom Fund nicht verlangt).
- **Tests:** `ResponsesApiSupportTest#inputItemOfAnUnknownTypeIsRejectedLikeUpstream` (Meldung nennt upstreams Text und `custom_tool_call`), als Schutz `#inputItemWithoutATypeStillMapsToAMessage` (vorher und nachher grün).
- **Rot-Grün-Nachweis:** Ohne den Fix rot („Expected java.lang.IllegalArgumentException to be thrown, but nothing was thrown.“), mit dem Fix grün (`ResponsesApiSupportTest` 67/67, `ResponsesStreamTranslatorTest` 5/5, `OpenAiCompatServerHttpTest` 69/69). `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `beaffcc`

### F229 – java-llama.cpp: Abgeschnittene Responses-Antwort (`finish_reason` `length`) wird als `completed` gemeldet
- **Zeit:** 2026-10-20 23:33 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/ResponsesStreamTranslator.java` (`onChunk`, `end`), `ResponsesApiSupport.java` (`toResponsesResponse`, neu `isTruncated`, `markIncomplete`)
- **Schwere:** niedrig (ein Client, dessen Antwort an `max_output_tokens` abgeschnitten wurde, sah eine vollständige Antwort)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20 (Fund J5-7). Seit es die Responses-Route gibt; upstreams `to_json_oaicompat_resp` (`server-task.cpp:579`, `:688`, b11639) meldet ebenfalls immer `completed`, es ist also Parität mit upstream, keine Regression. Bewusst über upstream hinaus wie F107 (Anthropics `tool_choice` `none`); vLLM verfährt so (PR #24561).
- **Beschreibung:** `ResponsesStreamTranslator.end()` und `ResponsesApiSupport.toResponsesResponse` bauten die Antwort fest mit `status` `completed`, der Strom endete immer mit `response.completed`; `finish_reason` wurde nie gelesen. Die Übersetzung bildet `max_output_tokens` auf `max_tokens` ab, upstreams Chunks tragen bei `STOP_TYPE_LIMIT` `finish_reason` `"length"`. Die Responses-API definiert dafür `status` `incomplete`, `incomplete_details.reason` `max_output_tokens`, das Ereignis `response.incomplete` und den Nachrichtenstatus `incomplete` (openai-python `Response`, `ResponseIncompleteEvent`, `ResponseOutputMessage`).
- **Fix:** Der Translator merkt sich das letzte textuelle `finish_reason` (der angehängte reine Usage-Chunk mit `"choices":[]` überschreibt es nicht, `null` der Zwischenchunks ebenso wenig). Bei `"length"` sind die Antwort `incomplete` mit `incomplete_details: {"reason":"max_output_tokens"}`, das Nachrichten-Item (in `response.output_item.done` und im `output`) `incomplete` und das letzte Ereignis `response.incomplete`; Reasoning- und `function_call`-Items bleiben `completed`, `completed_at` wird nicht gesetzt (die Hülle trug es nie). Dasselbe blockierend aus `choices[0].finish_reason`. Die Javadoc von `isTruncated` sagt, dass llama.cpps `"length"` auch einen vollen Kontext ohne Kontextverschiebung, `n_indent` und `t_max_predict_ms` umfasst; `max_output_tokens` ist der nächste der definierten Gründe, die Meldung also grob, nicht still.
- **Tests:** `ResponsesStreamTranslatorTest#maxTokensCutOffEndsWithResponseIncomplete` (Text, `length`, dann der reine Usage-Chunk; `response.incomplete`, kein `response.completed`, Status und Grund, Nachrichten-Item `incomplete`), `#aStoppedAnswerStaysCompleted`; `ResponsesApiSupportTest#responseCutOffAtTheTokenLimitIsIncomplete`, `#responseWrapsOutputMessageWithOutputTextAndUsage` prüft zusätzlich, dass eine `stop`-Antwort kein `incomplete_details` trägt. Die modellgestützten `OpenAiCompatServerIntegrationTest#responsesNonStreamingRoundTrip`/`#responsesStreamingRoundTrip` erwarteten mit `max_output_tokens` 16 `completed`; sie fragen jetzt eine Antwort ab, die in 4 Token nicht passt („Count from 1 to 100 …“), und erwarten `incomplete`/`response.incomplete` mit dem Grund – deterministisch statt davon abhängig, ob das Modell vor 16 Token aufhört.
- **Rot-Grün-Nachweis:** Ohne den Fix rot („Expected: is "incomplete" but: was "completed"“ und „Expected: a string containing "event: response.incomplete" but: was "event: response.output_text.done …"“); umgekehrt scheiterten nach dem Fix die unveränderten Integrationstests mit dem lokalen Winzmodell („Expected: is "completed" but: was "incomplete"“, der Beweis, dass der Fix den echten Server erreicht); mit dem Fix und den angepassten Integrationstests grün (`ResponsesApiSupportTest` 68/68, `ResponsesStreamTranslatorTest` 7/7, `OpenAiCompatServerHttpTest` 69/69, `OpenAiCompatServerIntegrationTest` 13/13 mit dem Winzmodell, zweimal). `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `ad8b0b1`

### F230 – java-llama.cpp: `/v1/rerank`: ein `top_n` als Gleitkommazahl gilt als „fehlt“, ein negatives (`-1.5`) umgeht die Abweisung
- **Zeit:** 2026-10-21 00:14 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OaiRerankSupport.java` (`readTopN`)
- **Schwere:** niedrig (still abweichend von upstream; die Abweisung aus F177 griff für die Gleitkomma-Schreibweise nicht)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20 (Fund J6-1). Die Hälfte „Gleitkommazahl = fehlt“ ist älter als der Branch (`return topN.isInt() ? topN.asInt() : -1;`); F177 setzte die Prüfung auf negative Werte hinter dieselbe `isInt()`-Bedingung, daher erreichte ein negativer Gleitkommawert sie nie.
- **Beschreibung:** `readTopN` gab für jeden Nicht-`IntNode` `-1` (= alle) zurück. Jackson liest `2.0` als `DoubleNode`, also behielt `"top_n": 2.0` alle Ergebnisse und `"top_n": -1.5` wurde nicht abgewiesen. Upstream liest `json_value(body, "top_n", (int)documents.size())` (`server-context.cpp:5760`, b11639) über nlohmanns `get<int>()`, das eine Gleitkommazahl per `static_cast` umwandelt: `2.0` und `2.7` behalten dort 2, `-1.0` lässt das `resize` in `format_response_rerank` scheitern. Ein Python-Client mit `top_n = len(docs) / 2` serialisiert `2.0`.
- **Fix:** `readTopN` liest jede JSON-Zahl im `int`-Bereich (`isNumber() && canConvertToInt()`, dann `asInt()`, das wie upstreams Cast zur Null hin abschneidet); ein negativer Wert ist danach in beiden Schreibweisen eine ungültige Anfrage (400). Zwei bewusste Abweichungen stehen in der Javadoc: ein Wert, der keine Zahl ist (Zeichenkette, Boolean – upstream wandelt `true` in 1), und eine Zahl außerhalb des `int`-Bereichs (upstream: der Cast läuft über) behalten alle Ergebnisse.
- **Tests:** `OaiRerankSupportTest#readTopNConvertsAFloatLikeUpstream` (`2.0` und `2.7` → 2), `#negativeFloatTopNIsAnInvalidRequest` (`-1.5`), `#readTopNKeepsAllForANonNumberOrAValueOutsideTheIntRange` (`"2"`, `true`, `4294967296` → -1, pinnt die dokumentierten Abweichungen).
- **Rot-Grün-Nachweis:** Ohne den Fix rot („Expected: is <2> but: was <-1>“ und „Expected java.lang.IllegalArgumentException to be thrown, but nothing was thrown.“), mit ihm grün (`OaiRerankSupportTest` 19/19, `OpenAiCompatServerHttpTest` 69/69); `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `5781ef5`

### F231 – java-llama.cpp: `/v1/rerank`: ein explizites `"documents": null` neben `texts` wird abgewiesen statt aus `texts` bedient
- **Zeit:** 2026-10-21 00:19 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OaiRerankSupport.java` (`readDocuments`)
- **Schwere:** niedrig (still abweichend von upstream; ein Client muss `null` neben `texts` senden)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20 (Fund J6-4). Seit F111 (`texts` von text-embeddings-inference akzeptiert); der Kommentar behauptete Parität mit upstream.
- **Beschreibung:** `readDocuments` wählte `request.has("documents") ? documents : texts`. Jacksons `has()` ist auch für einen expliziten `null`-Wert wahr, also nahm `{"documents":null,"texts":["a"]}` den `NullNode` und warf „'documents' (or TEI's 'texts') must be a non-empty array“ (400). Upstream liest `json_value(body, "documents", json_value(body, "texts", {}))` (`server-context.cpp:5753`, b11639), und `json_value` behandelt `null` wie „fehlt“ (`server-common.h:43-46`) – die Anfrage wird dort aus `texts` bedient.
- **Fix:** `hasNonNull("documents")`; Kommentar und Javadoc nennen den Rückfall bei `null`. Die Formatwahl in `toResponse` (`request.has("texts")`) bleibt, weil upstreams `body.contains("texts")` ein `null` ebenfalls mitzählt. Ein `documents` falschen Typs bleibt ein 400 (upstream fiele still auf `texts` zurück; das wäre ein stiller Rückfall).
- **Tests:** `OaiRerankSupportTest#readDocumentsFallsBackToTextsWhenDocumentsIsNull`, `#readDocumentsThrowsWhenDocumentsIsNullWithoutTexts`.
- **Rot-Grün-Nachweis:** Ohne den Fix rot („Unexpected exception thrown: java.lang.IllegalArgumentException: 'documents' (or TEI's 'texts') must be a non-empty array“), mit ihm grün (`OaiRerankSupportTest` 21/21, `OpenAiCompatServerHttpTest` 69/69); `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `893ede3`

### F232 – java-llama.cpp: `/v1/rerank`: ein nativer Antwortkörper, der kein JSON-Array ist, wird still zu einer leeren Ergebnisliste (200)
- **Zeit:** 2026-10-21 00:35 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/server/OaiRerankSupport.java` (`toTeiResponse`, `toOaiResponse`, neu `readNativeArray`), `LlamaModelBackend.java` (`rerank`)
- **Schwere:** niedrig (in der Praxis nicht erreichbar: der native Körper ist immer `safe_json_to_str(rerank_results_to_json(...))`, ein Fehler im nativen Aufruf wird vorher zur `LlamaException`; ein Vertrags- und Konsistenzfix, kein für Nutzer sichtbarer Fehler)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20 (Fund J6-5). Der Rückfall in `toOaiResponse` (samt `toOaiResponseOnMalformedNativeBodyYieldsEmptyResults`, der ihn festschrieb) ist älter als der Branch; F111 hat ihn in das neue `toTeiResponse` kopiert, ohne Test.
- **Beschreibung:** Beide Methoden fingen die `IOException` des Parsers ab und antworteten mit leeren Ergebnissen, ohne Log – für den Client ununterscheidbar von „nichts passt“. Zwei weitere stille Wege gingen nie durch den `catch`: Jackson liest einen leeren Körper als `MissingNode`, und ein Nicht-Array (`{}`) fiel durch `if (arr.isArray())`; beide ergaben ebenso `[]`. Der LangChain4j-Adapter (`LangChain4jMapping.parseRerankScores`) wirft im selben Fall.
- **Fix:** Ein Helfer `readNativeArray` für beide Methoden: ein Parserfehler, ein leerer Körper und ein Nicht-Array werfen eine `IOException` mit einem Ausschnitt des Körpers (höchstens 80 Zeichen) und der Ursache. `IOException` statt der im Fund vorgeschlagenen `IllegalStateException`, weil `OpenAiBackend.rerank` genau diesen Typ für einen Serverfehler deklariert (`LlamaModelBackend.rerank` deklariert ihn jetzt) und SpotBugs die Umwandlung in eine ungeprüfte Ausnahme ohne Unterdrückung nicht hinnimmt (`EXS_EXCEPTION_SOFTENING_NO_CONSTRAINTS`). `completeNonStreaming` antwortet darauf mit 500 `server_error` und einer WARN-Zeile. `json.RerankResponseParser.parse(String)` (dokumentierter Rückfall der typisierten API, PIT-gesichert) bleibt unverändert.
- **Tests:** `OaiRerankSupportTest#toOaiResponseRefusesANativeBodyThatIsNotAJsonArray` (ersetzt `#toOaiResponseOnMalformedNativeBodyYieldsEmptyResults`), `#toTeiResponseRefusesANativeBodyThatIsNotAJsonArray`; je `"not json"`, `""` und `"{}"`, und die Meldung zitiert den Körper.
- **Rot-Grün-Nachweis:** Ohne den Fix rot („not json ==> Expected java.io.IOException to be thrown, but nothing was thrown.“, in beiden Tests); eine Zwischenfassung, die nur im `catch` warf, blieb für den leeren Körper rot („Expected java.lang.IllegalStateException to be thrown, but nothing was thrown.“) – die zusätzlichen Fälle pinnen also den ganzen Fix. Mit dem Fix grün (`OaiRerankSupportTest` 22/22, `OpenAiCompatServerHttpTest` 69/69, `LlamaArchitectureTest` 13/13); `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `5c5a892`

### F233 – java-llama.cpp: ein leeres oder unparsbares Tool-Schema geht als String an llama.cpp statt in Java abgewiesen zu werden
- **Zeit:** 2026-10-21 01:05 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/parameters/ChatRequest.java` (`appendTool`, `buildToolsJson`, neu `parseSchemaOrThrow`)
- **Schwere:** niedrig (die Anfrage scheiterte laut, aber spät und ohne Werkzeugnamen; nur ohne Grammatik still)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20 (Fund J8-2). Der String-Rückfall für unparsbaren Text ist älter als der Branch; F05 schickte ein Schema mit nachfolgendem Text in denselben Rückfall, F55 ein leeres oder blankes Schema (vorher `"parameters": null`, das upstream als „keine Parameter“ las).
- **Beschreibung:** `buildToolsJson` schrieb ein Schema, das kein genau ein JSON-Wert war, als JSON-String unter `parameters`. Upstream (b11639) übernimmt den String als Schema (`common_chat_tools_parse_oaicompat`, `common/chat.cpp:580`; `common_chat_tool_parameters` ersetzt nur `null` und ein leeres Objekt, Zeile 546-554), und jeder Parser, der eine Tool-Grammatik baut, scheitert in `build_node` mit „schema must be an object“ (`common/json-schema.cpp:228`); `parse_oai_chat_params` macht daraus eine `InvalidRequestException`, deren Text weder das Werkzeug noch die Java-Ursache nennt. Der `force_pure_content`-Pfad (`chat.cpp:1375-1388`) rendert `"parameters":""` ohne Fehler in den Prompt. `JsonParameters.parseOrThrow` weist dieselbe Eingabeklasse mit einer `IllegalArgumentException` ab, der Branch behandelte denselben Fehler also zweifach.
- **Fix:** `appendTool` liest das Schema mit dem strikten Reader (`FAIL_ON_TRAILING_TOKENS`) und wirft für einen Parserfehler und für `MissingNode` (leer, blank) `IllegalArgumentException("tool '<name>': parameters must be exactly one well-formed JSON value, but was: <Ausschnitt>")`, Ausschnitt höchstens 80 Zeichen wie in `JsonParameters`, Ursache verkettet. `buildToolsJson` setzt nur noch den geparsten Knoten; beide String-Rückfälle sind weg. Ein Werkzeug ohne Parameter deklariert `{"type":"object","properties":{}}` (so wie `LangChain4jMapping.EMPTY_PARAMETERS_SCHEMA`), die Javadoc von `appendTool` sagt das. Nicht übernommen: `null` oder ein leeres Objekt für `""` einzusetzen (der zweite Vorschlag des Funds) wäre ein stiller Rückfall. SpotBugs meldet die Umwandlung der `IOException` (`EXS_EXCEPTION_SOFTENING_NO_CONSTRAINTS`); unterdrückt mit Begründung neben der gleichartigen von `JsonParameters.parseOrThrow`.
- **Warum `chat_template_kwargs` anders bleibt:** `ParameterJsonSerializer.buildRawValueObject` (F203) schreibt einen Wert, der kein JSON ist, weiter als String-Literal. Das ist dort dokumentiert und ein gültiger Wert: eine Template-Variable darf eine Zeichenkette sein (etwa `reasoning_effort`), upstream reicht sie an das Template durch. Ein Tool-Schema muss dagegen ein Objekt sein, und einen String weist upstream ab; derselbe Rückfall erzeugt hier also nie eine gültige Anfrage.
- **Tests:** `ChatRequestTest$JsonHelpers#appendToolRejectsASchemaThatIsNotExactlyOneJsonValue` (`""`, `"  "`, `"not json{{{"`, `{"type":"object"} {"required":["x"]}`, prüft die ganze Meldung mit dem Werkzeugnamen), `#appendToolBoundsTheRejectedSchemaInTheMessage` (300 Zeichen → 80 plus `...`); `#buildToolsJsonEmbedsAValidSchemaAsAnObject` prüft jetzt die genaue Ausgabe statt eines `containsString`. Ersetzt `#buildToolsJsonKeepsAnEmptySchemaAsTheStringFallbackInsteadOfNull` (F55) und den schwachen `#buildToolsJsonDoesNotSilentlyTruncateASchemaWithTrailingContent` (F05, prüfte nur `not(containsString(...))`).
- **Rot-Grün-Nachweis:** Ohne den Fix rot („Expected java.lang.IllegalArgumentException to be thrown, but nothing was thrown.“, in allen vier Fällen und im Längentest), mit ihm grün (`ChatRequestTest` 28/28, `ChatResponseTest` 8/8, `LlamaArchitectureTest` 13/13, `LangChain4jMappingTest` 23/23); `spotless:apply`, `spotbugs:check` und `javadoc:javadoc` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `0a49c54`

### F234 – java-llama.cpp: `appendCsvValue` überschreibt einen leeren Vorwert still statt ihn zu ergänzen
- **Zeit:** 2026-10-21 01:14 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/parameters/CliParameters.java` (`appendCsvValue`), `ModelParameters.java` (Javadoc von `addLoraAdapter`, `addControlVector`)
- **Schwere:** niedrig (ein ungültiger, leerer Adapterpfad verschwand ohne Meldung)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20 (Fund J8-7). Eingeführt mit F188 (`appendCsvValue`); dessen Eintrag nennt keinen Grund für den `isEmpty()`-Zweig, und seine Tests nutzten nie einen leeren Wert.
- **Beschreibung:** `appendCsvValue` schrieb `present == null || present.isEmpty() ? field : present + "," + field`. Ein leerer Vorwert kann nur aus einem früheren `addLoraAdapter("")` oder `addControlVector("")` stammen (die skalierten Varianten schreiben immer `FNAME:SCALE`), und der nächste Aufruf überschrieb ihn: `addLoraAdapter("").addLoraAdapter("b.gguf")` ergab `--lora b.gguf`. In anderer Reihenfolge (`b.gguf,`) oder allein (`--lora ""`) kam der leere Pfad dagegen bei upstream an, dessen `parse_csv_row` (`common/arg.cpp:1369-1410`, b11639) leere Felder behält und dessen `--lora`-Handler daraus einen Adapter macht, der beim Laden laut scheitert. Für einen fehlenden Schlüssel ist `present` ohnehin `null`; der Zweig wirkte nur als dieser stille Verlust.
- **Fix:** `appendCsvValue` weist einen leeren Wert mit `IllegalArgumentException("<option> needs a non-empty file path")` ab, im Stil der Meldung von `scaledField` (F187), und hängt einen vorhandenen Wert immer an (`present == null ? field : present + "," + field`). Abweisen statt Behalten, weil ein Fehler beim Bauen, der die Option nennt, klarer ist als „failed to load lora adapter“ beim Laden, und weil so auch der einzelne leere Wert früh auffällt. Die Javadoc der beiden ungeskalierten Methoden nennt die Ausnahme.
- **Tests:** `ModelParametersTest#anEmptyAdapterPathIsRefusedInsteadOfDropped` (`addLoraAdapter("").addLoraAdapter("b.gguf")` und `addControlVector("")` werfen mit der genauen Meldung; ein abgewiesener Wert lässt einen früheren `a.gguf` unverändert).
- **Rot-Grün-Nachweis:** Ohne den Fix rot („Expected java.lang.IllegalArgumentException to be thrown, but nothing was thrown.“), mit ihm grün (`ModelParametersTest` 98/98, `ModelParametersExtendedTest` 151/151, `WireNameRegistryTest` 8/8); `spotless:apply`, `spotbugs:check` und `javadoc:javadoc` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `b8c3992`

### F235 – java-llama.cpp: `submitAllThenReceive` gibt bei einem `Error` nichts frei und bricht das Empfangen ab
- **Zeit:** 2026-10-21 01:27 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/BatchDispatch.java` (`submitAllThenReceive`), `llama/spotbugs-exclude.xml` (Kommentare zu `BatchDispatch`)
- **Schwere:** niedrig (nur über einen `Error` wie `OutOfMemoryError` oder `StackOverflowError` erreichbar; native Fehler kommen als `LlamaException`)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20 (Fund J9-1). Eingeführt mit F167 (`BatchDispatch`); F200 hat die Schwestermethode `runEachOnItsOwnThread` für `Error` festgezogen, diese Methode nicht.
- **Beschreibung:** Beide Schleifen von `submitAllThenReceive` fingen nur `RuntimeException`. Ein `Error` beim Einreichen (`LlamaModel.completeBatch`/`completeBatchWithStats` rendern im Submit-Lambda `req.withStream(false).toJson()`, ein großer Prompt kann dort `OutOfMemoryError` werfen) lief an der Freigabe vorbei: die schon eingereichten Aufgaben generierten weiter, ihre Reader blieben bis `close()` in `jctx->readers`. Ein `Error` beim Empfangen (etwa beim Parsen eines großen Ergebnisses) übersprang alle folgenden Empfänge, also genau das Reader-Leck, das der Kommentar „Keep receiving the rest so no sibling task is left unread“ ausschließt. Beides widersprach der Javadoc der Methode und der von `completeBatch`.
- **Fix:** Die Einreichschleife fängt `Throwable`, gibt die eingereichten Aufgaben frei (auch die innere Freigabe fängt `Throwable`, damit ein `Error` einer Freigabe die übrigen nicht überspringt; er hängt als unterdrückt an) und wirft dieselbe Instanz per präzisem Rethrow weiter, ohne `throws`-Klausel und ohne `instanceof`-Kette. Die Empfangsschleife hält ein `Throwable failure`, empfängt alle Aufgaben und wirft den ersten Fehler über `asRuntimeException` (eine `RuntimeException` unverändert, ein `Error` unverändert). Javadoc und die beiden SpotBugs-Kommentare sagen jetzt „ein `Error` eingeschlossen“.
- **Tests:** `BatchDispatchTest#submitAllThenReceive_failedSubmissionWithAnError_releasesTheTasksAlreadySubmitted` (die dritte Einreichung wirft einen `AssertionError`; dieselbe Instanz kommt an, 10 und 20 sind freigegeben), `#submitAllThenReceive_aReleaseThatThrowsAnError_doesNotStopTheOtherReleases` (die Freigabe von 10 wirft einen `AssertionError`; 20 wird trotzdem freigegeben, der `Error` hängt unterdrückt an der Einreichfehlermeldung), `#submitAllThenReceive_anErrorFromAReceive_stillReceivesTheOthersThenThrowsIt` (der Empfang von 1 wirft einen `AssertionError`; 1, 2, 3 werden empfangen, dieselbe Instanz kommt an).
- **Rot-Grün-Nachweis:** Ohne den Fix alle drei rot („Expected: iterable containing [<10>, <20>] but: no item was <10>“, „Unexpected exception type thrown, expected: <java.lang.IllegalStateException> but was: <java.lang.AssertionError>“, „Expected: iterable containing [<1>, <2>, <3>] but: no item was <2>“), mit ihm grün (`BatchDispatchTest` 14/14); `spotless:apply` und `spotbugs:check` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `edad4ca`

### F236 – java-llama.cpp: die Batch-Methoden verwerfen jeden Fehler nach dem ersten ohne Spur
- **Zeit:** 2026-10-21 01:35 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/BatchDispatch.java` (`submitAllThenReceive`, `runEachOnItsOwnThread`, neu `suppress`), `LlamaModel.java` (Javadoc von `completeBatch`, `chatBatch`)
- **Schwere:** niedrig (der Aufrufer sah von mehreren fehlgeschlagenen Anfragen eines Batches nur die erste)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20 (Fund J9-3). Das Verhalten ist älter als der Branch: `LlamaModel.completeBatch` hielt schon vor F167 nur den ersten `join()`-Fehler fest, F167 hat das unverändert nach `BatchDispatch` übernommen. Der Test `submitAllThenReceive_failedReceive_stillReceivesTheOthersThenThrowsTheFirstFailure` ließ die zweite Anfrage `"second"` werfen und prüfte nie, ob davon etwas ankommt.
- **Beschreibung:** Die Empfangsschleife von `submitAllThenReceive` und die Warteschleife von `runEachOnItsOwnThread` schrieben `if (failure == null) failure = e;` und verwarfen jeden späteren Fehler; scheiterte in einem Batch Anfrage 1 als ungültig und Anfrage 3 an der Kontextgröße, sah der Aufrufer nur die erste. Der Freigabepfad derselben Klasse hängte spätere Fehler dagegen mit `addSuppressed` an, ohne Identitätsprüfung: wirft eine Freigabe dieselbe Instanz wie die Einreichung (eine geteilte oder zwischengespeicherte Ausnahme), lehnt `Throwable.addSuppressed` sie mit `IllegalArgumentException("Self-suppression not permitted")` ab; die entkam der Schleife, die übrigen Aufgaben wurden nicht freigegeben, und der Aufrufer sah die IAE statt des Fehlers.
- **Fix:** Spätere Fehler hängen als unterdrückt am ersten, in beiden Schleifen (bei `runEachOnItsOwnThread` die ausgepackte Ursache der `ExecutionException`). Alle drei Stellen gehen über `suppress(failure, later)`, das dieselbe Instanz überspringt (Identität, weil `addSuppressed` genau sie ablehnt; `@SuppressWarnings("ReferenceEquality")` mit Begründung). Die Javadoc beider Methoden und von `completeBatch`/`chatBatch` nennt das.
- **Tests:** `BatchDispatchTest#submitAllThenReceive_failedReceive_stillReceivesTheOthersThenThrowsTheFirstFailure` (erweitert: `"second"` hängt unterdrückt am ersten), `#runEachOnItsOwnThread_laterFailuresAreSuppressedOnTheFirst` (der Fehler von Anfrage 1 kommt an, der von Anfrage 2 hängt daran), `#submitAllThenReceive_aReleaseThatThrowsTheSubmissionFailure_isNotSuppressedOnItself` (beide Freigaben laufen, dieselbe Instanz ohne unterdrückte kommt an), `#submitAllThenReceive_theSameFailureForTwoReceives_isNotSuppressedOnItself` und `#runEachOnItsOwnThread_theSameFailureForTwoRequests_isNotSuppressedOnItself` (dieselbe Instanz zweimal: sie kommt ohne unterdrückte an, alle Aufgaben werden empfangen).
- **Rot-Grün-Nachweis:** Ohne den Fix drei rot („Expected: is <1> but: was <0>“ für die Länge von `getSuppressed()`, „Expected: is [<java.lang.IllegalStateException: second>] but: was []“, „Unexpected exception type thrown, expected: <java.lang.IllegalStateException> but was: <java.lang.IllegalArgumentException>“ für die Freigabe); die beiden Identitätstests der Schleifen sind gegen den alten Code grün (dort gab es kein `addSuppressed`) und wurden gegen ein `suppress` ohne Identitätsprüfung rot gezeigt (dieselbe IAE-Meldung). Mit dem Fix grün (`BatchDispatchTest` 18/18); `spotless:apply`, `spotbugs:check` und `javadoc:javadoc` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `4b410d1`

### F237 – java-llama.cpp: `RpcServer` gibt den Einzelinstanz-Platz nach 30 s frei, obwohl der Server-Thread noch lebt, und meldet „could not listen“
- **Zeit:** 2026-10-21 01:42 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/RpcServer.java` (`start`, `stopAndJoin`, `close`, neu `releaseSlotShare`)
- **Schwere:** niedrig (nur wenn der Server-Thread den Stopp mehr als 30 s überlebt, etwa eine Geräteinitialisierung vor dem Bind, die länger als Start- und Stopp-Frist zusammen dauert)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20 (Fund J9-4). F30 hat den Stopp eines fehlgeschlagenen Starts wiederholt, bis der Thread endet, das Warten aber auf `STOP_TIMEOUT_MILLIS` begrenzt und den Fall danach nicht behandelt; `close()` hatte dieselbe Form schon vor dem Branch (`joinQuietly`, dann `ACTIVE.set(false)`).
- **Beschreibung:** `stopAndJoin` lief, bis der Thread endete oder 30 s vergangen waren, und gab nichts zurück. `start` warf danach immer „RPC server could not listen on … (is the port already in use?)“, und sein `finally` gab den Platz frei, auch wenn der Thread noch lebte. Patch 0015 setzt das Stopp-Flag erst beim Bind zurück, ein solcher Thread kann also danach noch binden und lauschen, mit dem prozessweiten ggml-rpc-Zustand (`g_rpc_server_wake`, `g_rpc_server_client`), während ein zweiter `start` auf dieselben Globals zugelassen wurde: genau der verwaiste Server, den F30 ausschließen sollte, ohne Logzeile und mit einer Meldung, die das Gegenteil sagt. `close()` gab den Platz nach seinem 30-s-Join ebenso frei.
- **Fix:** Der Platz hat jetzt zwei Halter, sobald der Thread läuft: den Thread selbst (er lässt in einem `finally` nach `serve()` los) und den Start bzw. danach das `RpcServer`-Objekt (sein `close()` lässt los); erst der letzte gibt ihn frei (`releaseSlotShare`). `stopAndJoin` gibt zurück, ob der Thread geendet hat; wenn nicht, wirft `start` `LlamaException("RPC server on <endpoint> did not come up and its thread did not end within <n> ms after the stop; RPC servers stay unavailable in this process until it ends")` und loggt denselben Text als Warnung. `close()` warnt in diesem Fall („RPC server thread on … is still alive … ms after the stop …“) und wirft nicht. Zwei Halter statt „der Thread gibt frei“, weil ein noch offenes `RpcServer`, dessen `close()` die Globals stoppt, sonst einen neueren Server beenden könnte, nachdem sein eigener Thread von selbst geendet hat. Die Stopp-Frist geht als zusätzlicher `long` durch die paketinterne `start`-Überladung und in `close()`, damit die Tests sie kurz wählen; die öffentlichen Fabriken nutzen weiter 30 s.
- **Tests:** `RpcServerLifecycleTest#aServerThatIgnoresTheStopIsReportedAndKeepsTheSlot` (ein Backend, dessen `serve` den Stopp ignoriert, Start-Frist 100 ms, Stopp-Frist 200 ms: die Meldung nennt den überlebenden Thread, ein zweiter Start wirft `IllegalStateException`, nach dem Ende des Threads ist der Platz frei), `#aCloseThatTheServerThreadOutlivesKeepsTheSlot` (dasselbe für `close()`), `#aServerThreadThatEndsOnItsOwnKeepsTheSlotUntilClose` (der Platz bleibt bis `close()` belegt, auch wenn der Thread schon geendet hat; gegen den alten Code grün, er pinnt die Zwei-Halter-Regel).
- **Rot-Grün-Nachweis:** Zuerst nur die Stopp-Frist als Parameter eingeführt (verhaltensneutral), dann rot: „Expected: a string containing "its thread did not end within 200 ms after the stop" but: was "RPC server could not listen on 127.0.0.1:50052 (is the port already in use?)"“ und „Expected java.lang.IllegalStateException to be thrown, but nothing was thrown.“; mit dem Fix grün (`RpcServerLifecycleTest` 18/18, `RpcServerTest` 9/9 gegen die native Bibliothek, `RpcServerOptionsTest` 9/9); `spotless:apply`, `spotbugs:check` und `javadoc:javadoc` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `2eb3fff`

### F238 – java-llama.cpp: `--offline` meldet eine vorhandene Modelldatei mit Nicht-ASCII-Namen als fehlend, wenn die JVM mit der POSIX-Locale läuft
- **Zeit:** 2026-10-21 01:52 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/loader/OfflineModelGuard.java` (`localFileExists`, neu `countsAsMissing`, `fileNameEncoding`, `jnuEncoding`)
- **Schwere:** niedrig (braucht zugleich eine JVM ohne `LANG`/mit POSIX-Locale, einen Nicht-ASCII-Modellpfad und `--offline`; das ist allerdings genau die abgeschottete Container-Umgebung, für die `--offline` da ist)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20 (Fund J10-1). Eingeführt mit F56: dessen Fix machte jede `InvalidPathException` zu „fehlt“ und hatte nur das NUL-Zeichen im Blick. Vor dem Branch kam eine rohe `InvalidPathException` an, unhandlich, aber wahr.
- **Beschreibung:** `localFileExists` schrieb `try { return Files.exists(Paths.get(model)); } catch (InvalidPathException e) { return false; }`, und die Javadoc begründete das mit „a path the file system cannot represent“. Unter Linux kodiert `UnixPath` den Namen aber mit `sun.jnu.encoding`, das der Locale folgt; mit `LC_ALL=C` ist das `ANSI_X3.4-1968`, und `Paths.get("…/größe.gguf")` wirft „Malformed input or input contains unmappable characters“, obwohl die Datei als UTF-8-Name auf der Platte liegt. Die native Seite hat diese Grenze nicht (`jllama.cpp` gibt das argv als `String.getBytes("UTF-8")` weiter, upstream öffnet mit `std::filesystem::u8path`), ohne `--offline` lädt dasselbe Modell also. Mit `--offline` warf der Wächter selbstsicher `ModelUnavailableException` „the configured model file does not exist“ für eine vorhandene Datei.
- **Fix:** Die Entscheidung liegt in der paketinternen, reinen Funktion `countsAsMissing(model, charset, parse)`, in dieser Reihenfolge: ein NUL-Zeichen ist „fehlt“ (F56 bleibt; nativer Code schnitte den Pfad dort ab, auch wenn der Name zugleich nicht kodierbar ist); sonst entscheidet `Files.exists(parse(model))`; lehnt `parse` den Namen mit `InvalidPathException` ab, ist er nur dann „fehlt“, wenn der Zeichensatz ihn kodieren kann (die JVM konnte ihn schreiben, das Dateisystem hat ihn abgewiesen); kann er es nicht, geht der Pfad an den nativen Lader, der eine fehlende Datei selbst meldet. Der Zeichensatz ist `sun.jnu.encoding`, aufgelöst wie im JDK (`Charset.defaultCharset()`, wenn die Eigenschaft fehlt oder keinen unterstützten Zeichensatz nennt; nicht hart „UTF-8“), und unter Windows UTF-8, weil `WindowsPath` den UTF-16-String parst und `sun.jnu.encoding` nie liest, dort also jede Ablehnung die des Dateisystems ist (Präzisierung des Code-Skeptikers). Die Javadoc von `check` nennt den Fall.
- **Tests:** `OfflineModelGuardTest` (neu, im `loader`-Paket, weil die Funktion paketintern ist): `countsAsMissing_aNameTheJvmEncodingCannotRepresent_isLeftToTheNativeLoader` (US-ASCII und ein Parser, der wie `UnixPath.encode` wirft: nicht „fehlt“), `…_aNameTheFileSystemRejects_isMissing`, `…_aNulCharacter_isMissing`, `…_aNulCharacterWinsOverAnUnencodableName`, `…_anExistingAndAMissingFile`, `fileNameEncoding_isTheJnuEncodingExceptOnWindows`, `jnuEncoding_isTheCharsetOfTheProperty`, `jnuEncoding_anAbsentOrUnknownNameFallsBackToTheDefaultCharset`; der F56-Test `LlamaModelOfflineTest#check_offlineAndUnrepresentablePath_throwsModelUnavailable` bleibt grün. Zusätzlich von Hand geprüft mit `LC_ALL=C` (`sun.jnu.encoding=ANSI_X3.4-1968`) und einer vorhandenen Datei `größe.gguf`: der alte Wächter warf `ModelUnavailableException`, der neue lässt durch; ein fehlender Nicht-ASCII-Name geht jetzt ebenfalls an den nativen Lader.
- **Rot-Grün-Nachweis:** Zuerst die Funktion mit der alten Semantik herausgezogen (jede `InvalidPathException` heißt „fehlt“, Zeichensatz unbeachtet), dann rot: `countsAsMissing_aNameTheJvmEncodingCannotRepresent_isLeftToTheNativeLoader` „Expected: is <false> but: was <true>“; mit dem Fix grün (`OfflineModelGuardTest` 8/8, `LlamaModelOfflineTest` 9/9, `LlamaArchitectureTest` 13/13); `spotless:apply`, `spotbugs:check` und `javadoc:javadoc` grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `8809fe4`

### F239 – java-llama.cpp: der Loader lässt die `.tmp`-Reste unterbrochener Kopien des alten Layouts für immer liegen
- **Zeit:** 2026-10-21 01:56 UTC
- **Datei:** `llama/src/main/java/net/ladenthin/llama/loader/LlamaLoader.java` (`isStaleExtraction`, neu `isOlderThanTheStaleAge`, `FLAT_LAYOUT_TEMP_COPY`; Javadoc von `cleanup`, `isFlatLayoutLibraryName`, `FLAT_LAYOUT_SUFFIXES`)
- **Schwere:** niedrig (Restmüll im Temp-Verzeichnis, nur nach einer abgebrochenen Kopie einer Version vor 6.0.0)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20 (Fund J10-2). Folge von F09, dessen „Bitte prüfen“ genau diesen Fall offen ließ; die Namen sind jetzt am Tag `v5.1.0` verifiziert.
- **Beschreibung:** Bis 5.1.0 entpackte `LlamaLoader` `System.mapLibraryName("jllama")` flach ins Temp-Verzeichnis (`libjllama.so`, `libjllama.dylib`, `jllama.dll`), unter macOS dazu `ggml-metal.metal`, jede Kopie über `Files.createTempFile(dir, name + ".", ".tmp")` (die Zufallsmitte ist eine Dezimalzahl) und ein Verschieben an den Zielnamen. Eine JVM, die mitten in der Kopie starb, hinterließ `jllama.dll.<Ziffern>.tmp` bzw. `ggml-metal.metal.<Ziffern>.tmp`. Der alte Präfixfilter räumte die weg, die Bibliotheksendungs-Regel von F09 behielt sie (`.tmp` ist keine Bibliotheksendung). Außerdem stimmten die Javadocs nicht: `cleanup()` versprach die flachen Dateien `jllama*`, `llama*`, `ggml*` und `isFlatLayoutLibraryName` „a native library (.so, .dll, .dylib, .jnilib)“, obwohl `libjllama.so`/`libjllama.dylib` nie ein Bereinigungspräfix trugen und von keiner Version entfernt wurden.
- **Fix:** `isStaleExtraction` erkennt im Zweig für reguläre Dateien die Endung `\.\d+\.tmp$` (genau die Form von `createTempFile`); bleibt davor ein Bibliotheksname des alten Layouts, wird die Datei entfernt, sobald sie `STALE_EXTRACTION_AGE` (10 min) alt ist, denn eine jüngere kann einer JVM vor 6.0.0 gehören, die gerade kopiert (6.x schreibt nie in die Wurzel des Temp-Verzeichnisses). Die Altersprüfung teilt sich `isOlderThanTheStaleAge` mit dem Verzeichniszweig. Die reinen Bibliotheksnamen gehen weiter sofort. Die drei Javadocs nennen die verifizierten Namen. `testShouldCleanPathGgmlPrefix` prüft jetzt einen echten Namen des alten Layouts (`ggml-metal.metal.4711.tmp`) und sagt, dass es nur das Präfix-Prädikat prüft.
- **Tests:** `LlamaLoaderTest#staleExtractionRemovesAnInterruptedCopyOfTheOldLayoutOnceItIsOld` (alt: `jllama.dll.4711.tmp` und `ggml-metal.metal.<lange Zahl>.tmp` gehen; nicht: dieselbe Form ohne Alter, `somefile.so.4711.tmp` ohne Präfix, `jllama.gguf.4711.tmp` ohne Bibliotheksnamen, `jllama.dll.tmp` ohne Ziffern, ein Verzeichnis `jllama.dll.4713.tmp`).
- **Rot-Grün-Nachweis:** Ohne den Fix rot („expected: <true> but was: <false>“ für `jllama.dll.4711.tmp`), mit ihm grün (`LlamaLoaderTest` 64/64); `spotless:apply`, `spotbugs:check` und `javadoc:javadoc` grün. Die CHANGELOG-Zeile zu F09 („removes regular files with a library suffix … nothing else“) ist damit unvollständig; sie bleibt für den CHANGELOG-Durchgang.
- **Status:** gefixt
- **Commit:** java-llama.cpp `4205d20`

### F240 – java-llama.cpp: Patch 0021 hält einen nie benutzten Slot für einen Adapterwechsel und wirft dessen geladenen Prompt weg
- **Zeit:** 2026-10-21 02:22 UTC
- **Datei:** `llama/patches/0021-server-clear-the-slot-cache-when-the-server-adapters-change.patch` (zweiter Hunk in `init()` von `tools/server/server-context.cpp`, Kopftext), `llama/src/test/cpp/test_lora_cache.cpp`, `docs/patches.md`, `CLAUDE.md`
- **Schwere:** niedrig (konservativ: kein falsches Ergebnis, nur ein verlorener Cache-Treffer und eine irreführende Trace-Zeile)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20 (Fund J12-2). Folge von F67 (Patch 0021), dessen `else`-Zweig den Vergleich `params_base.lora_adapters` gegen `slot.lora` einführte.
- **Beschreibung:** `server_slot::lora` beginnt leer, und `init()` setzt es nie. Auf einem mit `--lora` gestarteten Server vergleicht der `else`-Zweig von Patch 0021 deshalb bei der ersten Aufgabe jedes Slots `[]` mit `[Adapter]`: `lora_should_clear_cache([], next)` ist für eine gewöhnliche LoRA wahr, der Slot protokolliert „clearing cache for lora change. 0 loras -> 1 loras“ und ruft `slot.prompt.clear()` auf. Auf einem leeren Slot kostet das nichts; hält der nie benutzte Slot aber schon einen Prompt -- per `SLOT_RESTORE` wiederhergestellt oder von `get_available_slot()` aus dem RAM-Prompt-Cache geladen, beides unter den Adaptern des Servers berechnet --, geht dieser Zustand verloren und der ganze Prompt wird neu verarbeitet. Vor dem Patch behielt der bloße `else`-Zweig ihn; der Satz des Patch-Kopfs „Equal sets keep the cache exactly as before“ galt für die erste Aufgabe eines Slots also nicht. Upstreams Zweig für Anfragen mit eigenen Adaptern hat dieselbe Asymmetrie, wenn eine Anfrage die Adapter des Servers nennt.
- **Fix:** Ein zweiter Hunk in Patch 0021 setzt in der Slot-Schleife von `init()` neben `slot.id = i;` `slot.lora = params_base.lora_adapters;`, sodass `slot.lora` von Anfang an „die Adapter, zu denen der Zustand des Slots gehört“ bedeutet. Ein `SET_LORA` vor der ersten Aufgabe eines Slots ist weiter ein Wechsel und leert ihn (auf einem leeren Slot folgenlos); `slot.lora` wird sonst nur von `common_set_adapter_lora` nach dem Start und von `can_batch_with` gelesen, die Initialisierung ist dort neutral. Der Patch-Kopf beschreibt den Hunk und hat je Hunk einen Drop-Check; `docs/patches.md` neu erzeugt, die CLAUDE.md-Zeilen von 0021 und `test_lora_cache.cpp` nachgezogen (646 Tests). Der ganze Patch-Satz ist auf die unveränderte Quelle b11639 (`a0e6dc738`) angewandt und ergibt für `server-context.cpp` genau den Diff des Build-Baums; `check-patches.py`: 14 Patches, 97 Hunks, 0 Probleme.
- **Tests:** `LoraCacheFreshSlot.AStateRestoredIntoANeverUsedSlotIsKept` (neue Fixture: zufälliges Modell und zufällige LoRA, `-np 2`, `--cache-ram 0`, `--slot-save-path`; Anfrage auf Slot 0, Slot 0 speichern, in den nie benutzten Slot 1 wiederherstellen, dieselbe Anfrage auf Slot 1: `timings.prompt_n` ≤ 2 und dieselbe Antwort wie auf Slot 0). Der vom Reviewer vorgeschlagene Test (`-np 2`, LRU, zweimal derselbe Prompt) wäre leer gewesen, weil der erste Zustand dabei nie in den RAM-Cache gelangt; deshalb der Weg über Speichern/Wiederherstellen, den der Praxis-Skeptiker beschrieben hat. Alle 646 C++-Tests grün.
- **Rot-Grün-Nachweis:** Ohne den `init()`-Hunk rot („Expected: (prompt_n) <= (2), actual: 77 vs 2“, `cache_n` 0: der wiederhergestellte Prompt war geleert), mit ihm grün (`LoraCache*` 5/5).
- **Status:** gefixt
- **Commit:** java-llama.cpp `166a267`

### F241 – java-llama.cpp: `jllama_rpc_server_stop`/`FfmRpcServer.stop` verlieren einen Stopp vor dem Bind; der Server lauscht weiter
- **Zeit:** 2026-10-21 02:35 UTC
- **Datei:** `llama/src/main/cpp/jllama_api.cpp` (`jllama_rpc_server_serve`, `jllama_rpc_server_stop`, neu `rpc_serving_scope`), `llama/src/main/cpp/jllama_api.h`, `llama/src/main/cpp/rpc_support.hpp` (Test-Seam), `llama-ffm/src/main/java/net/ladenthin/llama/ffm/FfmRpcServer.java` (Javadoc), `llama/src/main/java/net/ladenthin/llama/RpcServer.java` (Kommentar von `stopAndJoin`), `CLAUDE.md`
- **Schwere:** niedrig bis mittel (verwaister Listener, auf einem Netzwerk-Host ein offener Port; nur über die C-API und FFM, der JNI-`RpcServer` hatte die Wiederholung von F30)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20 (Fund XJ-1). F30 hat den verlorenen Stopp vor dem Bind nur in `RpcServer.stopAndJoin` (Java) behoben; die C-API und `FfmRpcServer` rufen dieselbe native Funktion einmal auf, und `FfmRpcServer.serve` empfahl das Abfragen von `isListening()` „as RpcServer does“.
- **Beschreibung:** Patch 0015 setzt in `ggml_backend_rpc_start_server` nach dem Binden `g_rpc_server_stop = false`. Ein `ggml_backend_rpc_stop_server()`, das davor ankommt (Geräteinitialisierung, Transport-Start), setzt das Flag, findet noch keine Weck-Adresse und wird beim Bind gelöscht: der Server lauscht weiter, bis ihn jemand erneut stoppt. `jllama_rpc_server_stop()` war ein einmaliger Durchreicher („no-op when none runs“), `FfmRpcServer.stop()` ebenso; ein FFM- oder C-Aufrufer, der nach einem Timeout einmal stoppt, glaubte den Server beendet. Das Zurücksetzen beim Bind selbst ist tragend: ohne es würde ein liegengebliebener Stopp (ohne laufenden Server spezifiziert als No-op) den nächsten Serve sofort beenden.
- **Fix:** Die C-API führt den Zustand selbst (Variante (b) des Praxis-Skeptikers, Patch 0015 bleibt unverändert): `jllama_rpc_server_serve` markiert sich für seine ganze Dauer als laufend (`rpc_serving_scope`, Zähler unter Mutex, dazu ein `thread_local` für den Serving-Thread); ein Stopp während eines laufenden Serve wird als Anforderung gehalten. Serve startet den ggml-Server nicht mehr, wenn die Anforderung vor dem Aufruf vorliegt, und gibt OK zurück; ein Stopp von einem anderen Thread wiederholt `ggml_backend_rpc_stop_server()` alle 50 ms, bis Serve zurückgekehrt ist, und gibt erst dann OK zurück, nach 30 s `JLLAMA_STATUS_ERROR` („the RPC server did not stop within 30 s of the stop request“). Ein Stopp auf dem Serving-Thread selbst (aus einem Callback vor dem Start) kehrt sofort zurück. Ohne laufenden Serve bleibt der Stopp ein No-op, und die Anforderung endet mit dem letzten Serve, erreicht also nie den nächsten. Ein Stopp, bevor der Serving-Thread `jllama_rpc_server_serve` überhaupt betreten hat, ist von „keiner läuft“ nicht zu unterscheiden und bleibt verloren; das steht jetzt in `jllama_api.h` und im Javadoc von `FfmRpcServer.stop`/`serve` (wiederholen, bis Serve zurückgekehrt ist, wie `RpcServer`). Test-Seam: `jllama::rpc::set_serve_hook_for_testing()` (in `rpc_support.hpp` deklariert, nicht exportiert), aufgerufen nach der Geräteauswahl und vor dem ggml-Aufruf. Folge für den JNI-Pfad: `RpcServer.close()` und `stopAndJoin` warten jetzt schon im nativen Stopp; im pathologischen Fall, dass Serve 30 s nach dem Stopp nicht zurückkehrt, wirft der native Stopp eine `LlamaException`, statt dass `close()` nur warnt.
- **Tests:** `CApiRpc.AStopBeforeTheBindFromTheServingThreadEndsTheServe` (Stopp aus dem Seam auf dem Serving-Thread), `CApiRpc.AStopBeforeTheBindFromAnotherThreadReturnsOnlyOnceTheServeHasEnded` (Serve im Seam festgehalten, Stopp von einem anderen Thread darf nicht vor dem Serve zurückkehren, danach endet der Serve), `CApiRpc.AStopWhileNoServeRunsDoesNotStopTheNextOne` (Schutz gegen einen klebenden Stopp), `FfmRpcServerTest.aStopWhileNoServeRunsDoesNotStopTheNextServeAndAStopEndsIt` (FFM über Loopback: ein vorheriger Stopp ohne Server hält den nächsten Serve nicht ab, ein Stopp beendet ihn; der Fall vor dem Bind braucht den Seam und ist in C++ gepinnt). Daneben grün: alle 649 C++-Tests, `RpcServerTest`, `RpcServerLifecycleTest`, `RpcServerOptionsTest`, `RpcIntegrationTest` (Winzmodell), `llama-ffm verify` 36/36.
- **Rot-Grün-Nachweis:** Mit dem Seam, aber ohne den Fix rot: „Value of: run.returned_within(std::chrono::seconds(10)) Actual: false Expected: true — serve went on after a stop issued before the bind“ und „Value of: jllama_rpc_server_listening() Actual: true Expected: false“ (erster Test); „Value of: stop_changed.wait_for(lock, std::chrono::milliseconds(300), …) Actual: true Expected: false — the stop returned while serve had not even started the server“ (zweiter Test). Mit dem Fix grün (`CApiRpc*`/`Rpc*` 32/32, dreimal). Der dritte und der FFM-Test sind Regressionsschutz und vorher wie nachher grün.
- **Nachtrag (Branch-Review 2026-10-20):** Bei der Prüfung nachgezogen: das Javadoc von `RpcServer.close()` sagte weiter, nach 30 s werde nur gewarnt; seit diesem Fix wirft der native Stopp in dem Fall eine `LlamaException` aus `close()`. Das Javadoc beschreibt jetzt beide Fälle und hat ein `@throws`.
- **Status:** gefixt
- **Commit:** java-llama.cpp `1047673`

### F242 – java-llama.cpp: `llama-langchain4j` meldet `onCompleteToolCall` eines parallelen Werkzeugaufrufs erst am Stream-Ende, nach den Teilstücken des nächsten
- **Zeit:** 2026-10-21 02:47 UTC
- **Datei:** `llama-langchain4j/src/main/java/net/ladenthin/llama/langchain4j/StreamingChunkAssembler.java` (`acceptToolCallFragment`, neu `completeCallsBelow`, `complete`)
- **Schwere:** niedrig (Reihenfolge der Ereignisse; die Methoden sind in langchain4j `@Experimental`, kein falsches Endergebnis)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20 (Fund J14-5). Vorbestehend seit dem gestreamten Werkzeugaufruf in `llama-langchain4j` (5.0.6); F128 hat am selben Zustandsautomaten nur die leeren Teilstücke abgefangen.
- **Beschreibung:** `onCompleteToolCall` wurde nur in `complete()` gesendet, in einer Schleife über alle Aufrufe. Bei zwei parallelen Aufrufen sah ein Handler deshalb `partial(0)…, partial(1)…, complete(0), complete(1)`. langchain4j 1.22.0 beschreibt `onCompleteToolCall` als „Invoked when the model has finished streaming a single tool call“, und seine eigene OpenAI-Anbindung (`ChatCompletionEventDispatcher`) schließt einen Aufruf ab, sobald ein Fragment mit einem anderen Index eintrifft, also vor dem ersten Teilstück des nächsten. Ein Handler, der ein Werkzeug ausführt, sobald es vollständig ist, wartete so bis zum Ende des Streams. Upstream macht den frühen Abschluss sicher: `common_chat_msg_diff::compute_diffs` (b11639, `common/chat.cpp`) erweitert nur den letzten Aufruf der vorigen Nachricht und hängt neue Indizes aufsteigend an; ein Aufruf N bekommt nach dem Start von N+1 nichts mehr. Der vorhandene Test `assemblesParallelToolCallsByIndex` prüfte nur die Anzahl, und `RecordingHandler` hielt je Ereignistyp eine eigene Liste, sah die Reihenfolge also nicht.
- **Fix:** Fassung des Praxis-Skeptikers. Je Aufruf ein Merker `completed`, im Assembler `highestIndex`. Trifft ein Fragment mit einem **höheren** Index ein (ein niedrigerer stößt nichts an), werden vorher alle offenen niedrigeren Aufrufe, die schon einen Namen haben, in Index-Reihenfolge abgeschlossen (`onCompleteToolCall`); ein Aufruf ohne Namen bleibt offen und wird von `complete()` gemeldet, damit nie ein `CompleteToolCall` mit leerem Namen ausgeht (der Fall von `filter_tool_calls`/F128, in dem der Name erst später kommt). Ein Fragment, das einen schon abgeschlossenen Aufruf ändert (nicht leere Argumente, ein anderer Name oder eine andere Id), ist eine Verletzung des nativen Vertrags und wirft `IllegalStateException` mit dem Index, die wie ein unparsbarer Chunk über `onError` ankommt; still zusammengeführt würde die endgültige `ToolExecutionRequest` von dem `CompleteToolCall` abweichen, den der Handler schon hat. `complete()` meldet nur die noch offenen Aufrufe, baut die `toolExecutionRequests` aber weiter aus allen in Index-Reihenfolge. Klassen-Javadoc nachgezogen.
- **Tests:** `StreamingChunkAssemblerTest`: `RecordingHandler` führt zusätzlich eine geordnete Liste `toolCallEvents`; `assemblesParallelToolCallsByIndex` prüft jetzt die Reihenfolge (beide Aufrufe in einem Chunk); neu `completesEachToolCallAsSoonAsTheNextOneStarts` (drei Chunks, Ereignisse genau `partial:0, complete:0, partial:1, complete:1`, die `CompleteToolCall`-Nutzlasten gleich den endgültigen Requests), `defersCompletionOfAToolCallThatHasNoNameYetWhenTheNextOneStarts` (Aufruf 0 ohne Namen beim Start von 1: `partial:1, complete:0, complete:1`, Name aus dem späten Fragment; Regressionsschutz, vorher wie nachher grün), `argumentsForAToolCallThatIsAlreadyCompleteFailTheStream`. Modul `llama-langchain4j` 58/0/0/7 grün.
- **Rot-Grün-Nachweis:** Gegen den unveränderten Code rot: `assemblesParallelToolCallsByIndex` und `completesEachToolCallAsSoonAsTheNextOneStarts` mit „Expected: iterable containing ["partial:0", "complete:0", "partial:1", "complete:1"] but: item 1: was "partial:1"“, `argumentsForAToolCallThatIsAlreadyCompleteFailTheStream` mit „Expected java.lang.IllegalStateException to be thrown, but nothing was thrown.“; mit dem Fix alle 12 Tests der Klasse grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `b90de3c`

### F243 – java-llama.cpp: der Atmosphere-Agent nennt in jeder Antwort seines In-Process-Servers den absoluten Modellpfad statt der Modell-Id
- **Zeit:** 2026-10-21 02:50 UTC
- **Datei:** `llama-atmosphere-agent/src/main/java/net/ladenthin/llama/atmosphere/ModelEndpoint.java` (`modelParameters`), `llama-atmosphere-agent/src/main/java/net/ladenthin/llama/atmosphere/AgentOptions.java` (`parse`, neu `requireAliasSafeModelId`), `llama/src/main/java/net/ladenthin/llama/parameters/ModelParameters.java` (Javadoc von `setAlias`)
- **Schwere:** niedrig (Loopback-Server; ein falscher `model`-Wert und ein lokaler Pfad in den Antworten, kein falsches Ergebnis)
- **Historie:** Gefunden beim Review des Branches am 2026-10-20 (Fund XJ-2). Folge von F185 (jede native OpenAI-Antwort trägt upstreams Modellnamen, vorher ""); `15f857a` hat dieselbe Abweichung für `OpenAiServerCli` behoben, indem die CLI immer `--alias` setzt, und F201 die Modell-Id dort auf Komma/Randleerzeichen geprüft. Der CHANGELOG sagt, ein Aufrufer mit eigenem `LlamaModel` müsse `setAlias` selbst setzen; der Agent ist ein solcher Aufrufer und tat es nicht.
- **Beschreibung:** `ModelEndpoint.modelParameters` lud das In-Process-Modell ohne `setAlias`, `open()` bot es mit `OpenAiServerConfig.modelId(options.getModelId())` an (Vorgabe `local-model`). Upstream b11639 (`server-context.cpp`, Modellname) nimmt ohne Alias `params_base.model.get_name()`, und das ist für eine lokale Datei der Pfad (`common.h`, `common_params_model::get_name`). `OpenAiCompatServer` reicht die native Antwort unverändert weiter. Also listete `/v1/models` `local-model`, während jede `/v1/chat/completions`-Antwort und jedes Stück des Streams z. B. `/home/user/models/qwen.gguf` nannte. `AgentOptions` nahm `--model-id` ungeprüft; da `--alias` an Kommas trennt und jedes Stück trimmt, hätte ein bloßes `setAlias` für `a,b` oder ` x` dieselbe Abweichung behalten.
- **Fix:** `modelParameters` setzt `setAlias(options.getModelId())`, die Regel von `OpenAiServerCli.toModelParameters`. `AgentOptions.parse` lehnt mit `--model` eine `--model-id` mit Komma, eine leere und eine mit Randleerzeichen ab, wie `OpenAiServerCli.modelIdValue`; mit `--base-url` bleibt die Id ungeprüft, weil sie dort ein Modell eines fremden Servers benennt und kein Alias wird (Abweichung vom Brief, der die Prüfung allgemein verlangt). Das Javadoc von `ModelParameters.setAlias` sagte, ohne Alias nehme llama.cpp den eigenen Namen oder den Dateinamen; es nennt jetzt das Repository bzw. den Pfad wie an `--model` übergeben und dass jedes Alias getrimmt wird.
- **Tests:** `LocalAgentTest.inProcessModelAnswersUnderTheModelIdItIsListedWith` (`--alias` = `AgentOptions.DEFAULT_MODEL_ID` ohne, `x` mit `--model-id x`), `AgentOptionsTest.modelIdThatLlamaCppsAliasWouldSplitOrTrimIsRejectedForAnInProcessModel` (`a,b`, ` x`, `x `, `""`), `AgentOptionsTest.modelIdForARemoteEndpointIsSentUnchanged` (`a,b` mit `--base-url` bleibt). `LocalAgentTest` 28/28, `AgentOptionsTest` 17/17 grün; Kern-Javadoc ohne Fehler.
- **Rot-Grün-Nachweis:** Gegen den unveränderten Code rot: „Expected: a collection containing "--alias" but: mismatches were: [was "", was "--fit", was "off", was "--model", was "m.gguf", …]“ und „Expected java.lang.IllegalArgumentException to be thrown, but nothing was thrown.“; mit dem Fix beide grün.
- **Status:** gefixt
- **Commit:** java-llama.cpp `f10aa6a`

## Offene Verdachte ohne Fund-ID

- *(geschlossen 2026-10-19, ohne Fund-ID, Entscheidung des Nutzers: bewusst nicht ändern)* Continue entscheidet die FIM-Fähigkeit eines Ollama-Modells an `{{ .Suffix }}` im `template` von `/api/show`; wir liefern `""`. Ein Platzhalter-Template wäre erfunden (der Server kennt das Prompt-Format des Modells nicht) und für die meisten Modelle falsch. Continue nutzt dann eigene Vorlagen; für die FIM-Token des Modells gibt es `POST /infill` (Continues `llama.cpp`-Provider). README-Hinweis ergänzt.
