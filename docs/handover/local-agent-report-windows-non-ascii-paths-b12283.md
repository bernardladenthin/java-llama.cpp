# Windows non-ASCII paths at b12283: local measurement of patches 0025–0030

The report covers two local runs on Windows, both on branch `claude/wonderful-tesla-1fkezz`:

- **First run, at `1e4e764a`** ("Upgrade llama.cpp from b12242 to b12283"): patches 0026–0028 and the
  Java layer's model load. These are the sections "Step 1" to "Step 4".
- **Second run, at `0d9468ac`**: 20 patches. It covers 0029, 0030 and the sliding-window half of
  0025, the open Java-layer points (`--log-file`, slots, the quantizer, the extraction directory), a
  test defect it found, and the tests added for it. This is the section
  ["Second run at 0d9468ac"](#second-run-at-0d9468ac).

No patch was changed in either run. The patch files moved out for the counter-checks were moved
back (`git status` shows no change under `llama/patches`).

"Measured" means run on this machine in this session. "Read" means taken from the source or from a
patch header without running it.

## Environment (measured)

| Item | Value |
|---|---|
| OS | Windows 11 Pro 10.0.26300 |
| `GetACP()` / `GetOEMCP()` (kernel32, from a PowerShell process) | **1252** / **850** |
| Registry `HKLM\...\Nls\CodePage` ACP / OEMCP | 1252 / 850 (so the "Beta: UTF-8 for worldwide language support" switch is off; it would set ACP to 65001) |
| `chcp` in this agent's shell | 65001. This is the console code page of the agent's terminal only, not the process's ANSI code page; `GetACP()` stays 1252 |
| Compiler | MSVC 19.44.35229.0 (VS 2022 Community, `vcvars64.bat`), Ninja Multi-Config |
| llama.cpp | b12283, commit `f3b407ca1bfeeecebf9e9ef19fcd6eed565bd70a` (configure: "fetched commit count 12283 = b12283") |
| JDK | Oracle JDK 21.0.9 installed; for step 4, Temurin 25.0.5+8 unpacked into the scratch directory, not installed |

## Step 1: static test build (measured)

```
.github\build.bat -G "Ninja Multi-Config" -DOS_NAME=Windows -DOS_ARCH=x86_64 -DBUILD_TESTING=ON
```

The command ran from a `vcvars64` cmd into a fresh `llama\build`; the old b12242 directory was
renamed to `llama\build.b12242-bak` first. The result was green (`BUILDBAT_RC=0`, 526/526 steps).
The build-info line:

```
patches 0002,0003,0006,0007,0008,0014,0015,0017,0018,0020,0021,0022,0023,0024,0025,0026,0027,0028, MSVC 19.44.35229.0, toolkit none
```

## Step 2: tests with every patch (measured)

```
llama\build\Release\jllama_test.exe --gtest_filter=CApiNonAsciiPaths.*:CApiSlotFiles.*
```

| Test | Result |
|---|---|
| CApiSlotFiles.ASlotFileIsNamedInsideTheSlotSaveDirectory | green |
| CApiSlotFiles.ASlotDirectoryWithNonAsciiCharactersHoldsTheWholeSlotFile | green |
| CApiSlotFiles.ASlotFileNameWithNonAsciiCharactersIsOneFileThatRestoresWhole | green |
| CApiNonAsciiPaths.TheModelFilesAreReadUnderANonAsciiPath | green |
| CApiNonAsciiPaths.TheMediaDirectoryIsCheckedAsUtf8 | green |
| CApiNonAsciiPaths.TheLogFileIsWrittenUnderItsNonAsciiName | green |
| CApiNonAsciiPaths.TheImatrixQuantizerAndTrainerReadAndWriteUnderANonAsciiPath | green |

`[  PASSED  ] 7 tests.`, exit code 0.

The green run also logs
`E gguf_init_from_reader: invalid magic characters: '????', expected 'GGUF'` four times. Those lines
come from the merge's loader, which tries the legacy `.dat` file as GGUF before it falls back to the
legacy reader (read). They are not a failure, and they appear in every variant below as well.

## Step 3: counter-check, one patch removed at a time (measured)

For each patch:

- The patch file was moved out of `llama/patches`.
- A fresh build directory (`llama\build-no00NN`) was configured with the same flags as step 1 and
  `-DGGML_CCACHE=OFF`, as `build.bat` passes.
- Only the target `jllama_test` was built (`cmake --build … --config Release --target jllama_test`).
- The patch file was moved back after the build. The build-info lines are marked
  `1e4e764a…-dirty` because the patch was missing during configure.
- The same gtest filter was run.

| Test | all patches | without 0026 | without 0027 | without 0028 |
|---|---|---|---|---|
| CApiSlotFiles.ASlotFileIsNamedInsideTheSlotSaveDirectory | green | green | green | green |
| CApiSlotFiles.ASlotDirectoryWithNonAsciiCharactersHoldsTheWholeSlotFile | green | green | green | green |
| CApiSlotFiles.ASlotFileNameWithNonAsciiCharactersIsOneFileThatRestoresWhole | green | green | green | green |
| CApiNonAsciiPaths.TheModelFilesAreReadUnderANonAsciiPath | green | green | green | green |
| CApiNonAsciiPaths.TheMediaDirectoryIsCheckedAsUtf8 | green | green | green | green |
| CApiNonAsciiPaths.TheLogFileIsWrittenUnderItsNonAsciiName | green | green | **red** | green |
| CApiNonAsciiPaths.TheImatrixQuantizerAndTrainerReadAndWriteUnderANonAsciiPath | green | **red** | green | **red** |

The patch lists the configure step reported:

- without 0026: `…,0024,0025,0027,0028`
- without 0027: `…,0024,0025,0026,0028`
- without 0028: `…,0024,0025,0026,0027`

**Finding:** the result is the one expected. Removing a patch turns exactly its own test red, and
every other test stays green. No test stays green without the patch it holds, so none of the three
tests is without effect on this machine (code page 1252).

### Failure output, verbatim

**Without 0026**, in the "a non-ASCII directory" shape:

```
E gguf_init_from_reader: invalid magic characters: '????', expected 'GGUF'
E common_imatrix_load_legacy: failed to open 
C:\Users\<user>\AppData\Local\Temp\jllama-c-api-utf8-tools-260408325\d-ä-日\f.imatrix.dat
X:\...\llama\src\test\cpp\test_c_api.cpp(3647): error: Expected equality of these values:
  jllama_imatrix_merge(one_legacy, 1, back.c_str(), false, &chunks_back)
    Which is: 1
  JLLAMA_STATUS_OK
    Which is: 0
could not read the importance matrix 'C:\Users\<user>\AppData\Local\Temp\jllama-c-api-utf8-tools-260408325\d-ä-日\f.imatrix.dat'
Google Test trace:
X:\...\llama\src\test\cpp\test_c_api.cpp(3614): a non-ASCII directory
```

The legacy file that this library's own writer produced just before is not found by upstream's
legacy reader: the narrow `std::ifstream(fname)` at `common/imatrix-loader.cpp:11` (read).

**Without 0027**, in both shapes:

```
X:\...\llama\src\test\cpp\test_c_api.cpp(3588): error: Value of: std::filesystem::exists(log, ec)
  Actual: false
Expected: true
no log file under the name that was asked for
Google Test trace:
X:\...\llama\src\test\cpp\test_c_api.cpp(3580): a non-ASCII directory
```

```
X:\...\llama\src\test\cpp\test_c_api.cpp(3588): error: Value of: std::filesystem::exists(log, ec)
  Actual: false
Expected: true
no log file under the name that was asked for
Google Test trace:
X:\...\llama\src\test\cpp\test_c_api.cpp(3580): a non-ASCII file name
```

The load and generation themselves succeed; only the file is missing under its real name. This is
the narrow `fopen(path, "w")` in `common/log.cpp` (read; measured at b12242 in the earlier report as
"no file in the dir shape, mangled name in the name shape").

**Without 0028**, in the "a non-ASCII directory" shape:

```
E gguf_init_from_reader: invalid magic characters: '????', expected 'GGUF'
E gguf_init_from_reader: invalid magic characters: '????', expected 'GGUF'
E llama_model_quantize: failed to quantize: ios_base::failbit set: iostream stream error
X:\...\llama\src\test\cpp\test_c_api.cpp(3662): error: Expected equality of these values:
  jllama_quantize_imatrix(utf8(in).c_str(), utf8(quantized).c_str(), 15, 1, false, utf8(imatrix).c_str())
    Which is: 1
  JLLAMA_STATUS_OK
    Which is: 0
Quantization of 'C:\Users\<user>\AppData\Local\Temp\jllama-c-api-utf8-tools-1349071104\d-ä-日\f.gguf' failed with code 1
Google Test trace:
X:\...\llama\src\test\cpp\test_c_api.cpp(3614): a non-ASCII directory
```

The output stream cannot be opened because the mis-decoded directory does not exist. That stream is
the narrow `std::ofstream(fname)` in `src/llama-quant.cpp` (read). Everything before line 3662 in
this shape passed, including the legacy round trip, because 0026 was in place.

The user name in the paths is replaced with `<user>`, and the repository prefix is shortened to `X:\...`.

### Limits of the counter-check

- **Without 0026 and without 0028:** the test fails with an `ASSERT_EQ` in its first iteration (the
  "a non-ASCII directory" shape), so the "a non-ASCII file name" shape of the same test did not run
  in these two variants. So for 0026 and 0028 only the directory shape is measured red. The
  name-only shape is not measured red: it would need a test change, and this task did not allow one.
- **Without 0027:** both shapes ran, and both are red (`EXPECT_*`, not `ASSERT_*`).
- **What the variant builds contain:** they built only `jllama_test`, not `jllama.dll`. The Java
  layer was not involved in step 3.

## Step 4: Java layer (measured)

Temurin 25 is not installed on this machine; only Oracle JDK 21.0.9 is. A Temurin 25 build was
downloaded instead and unpacked into the agent's scratch directory, without being installed. Only
`JAVA_HOME` and `PATH` of the Maven process point at it.

- **Build:** `OpenJDK25U-jdk_x64_windows_hotspot_25.0.5_8.zip`, 25.0.5+8 LTS.
- **Source:** the Adoptium API `v3/assets/latest/25/hotspot`, windows/x64/jdk.
- **Checksum:** SHA-256 `636bdc13dc649ddc7d5921c5c7d23cca6e44fd19d662b45cc9c02135a6c96c42`, equal to
  the checksum the API publishes.

The library under test is the step 1 build: `src/main/natives/.../Windows/x86_64/cpu/jllama.dll`, a
static library with all patches up to 0028.

**What overwrote the metadata first.** The configure steps of the three counter-check builds had
regenerated `jllama-build.txt` and `jllama-sbom.json` in that directory through `file(GENERATE)`.
The file then named the last variant: `patches=…,0026,0027` and `jllama.commit=…-dirty`. The DLL
itself was untouched: its timestamp, 08:42:31, is that of `build\Release\jllama_test.exe` from step 1.
A re-configure of `llama\build` restored both files: `patches=…,0026,0027,0028` and
`jllama.commit=1e4e764a…` without `-dirty`. The runs below were repeated after that, with the same
result as before it.

So a configure into another build directory with a different patch set rewrites the provenance files
of the shared natives tree, even when only `--target jllama_test` is built afterwards. Locally this
is easy to miss (measured).

### Maven

```
mvn -B -f llama/pom.xml test -Dtest=TextToSpeechBackendTest,NativeLibraryLoadSmokeTest
```

| Test class | Run | Failures | Errors | Skipped |
|---|---|---|---|---|
| `loader.NativeLibraryLoadSmokeTest` | 7 | 0 | 0 | 0 |
| `server.TextToSpeechBackendTest` | 9 | 0 | 0 | 1 |

`BUILD SUCCESS`.

The skipped test is `theReferenceFileIsReadableByItsOwnerOnly`, through
`assumeTrue(FileSystems.getDefault().supportedFileAttributeViews().contains("posix"))`. It is
POSIX-only by design, and Windows has no `posix` view.

### `LlamaModel` under `C:\Temp\Prüfung\`

The probe is a small program run with the module's runtime classpath plus `target\classes` and
`src\main\natives`. It loads `new LlamaModel(new ModelParameters().setModel(path).setGpuLayers(0).setCtxSize(256))`
and runs one `complete(…withNPredict(8))`. The model is `AMD-Llama-135m-code.Q2_K.gguf`, copied
three times.

The paths are built inside Java from `\u` escapes, not passed on the command line. The reason is
that `sun.jnu.encoding` is `Cp1252`, as measured, and the Java launcher reads the command line in
that code page, so `日` would arrive as `?`. That would be a test of the launcher, not of this library.

```
file.encoding=UTF-8 sun.jnu.encoding=Cp1252 native.encoding=Cp1252 defaultCharset=UTF-8
[jllama] native backend 'cpu' loaded from C:\Users\<user>\AppData\Local\Temp\jllama-backend-cpu-1a1c98ef31f-c4be00 with no GPU module; 2 file(s) in place for ggml to load
== C:\Temp\Prüfung\model.gguf exists=true size=60255360
RESULT ok, completion= Parambig bet bet bet amount'. param
== C:\Temp\Prüfung\日本\model.gguf exists=true size=60255360
RESULT ok, completion=double also bet Element Argument Net Net Net
== C:\Temp\Prüfung\m-ä-日.gguf exists=true size=60255360
RESULT ok, completion= Net nut net amount Net net net net
```

| Path | Load and completion |
|---|---|
| `C:\Temp\Prüfung\model.gguf` (cp1252-representable) | ok |
| `C:\Temp\Prüfung\日本\model.gguf` (CJK directory) | ok |
| `C:\Temp\Prüfung\m-ä-日.gguf` (non-ASCII file name) | ok |

The completions are nonsense because the model is a 135M Q2_K model; the test is that a load and
generation happened. So `-m` through the Java layer — Java `String` to UTF-8 through JNI to
`ggml_fopen` — works on code page 1252. The same holds for a path that no single-byte code page can
represent.

What this does not cover:

- the Java-side paths of the other options (`--log-file`, slots, the quantizer and so on): only
  their C API is measured, in step 2;
- the extraction directory under a non-ASCII `java.io.tmpdir` or user profile: here it was the
  ASCII path `C:\Users\<user>\AppData\Local\Temp`.

## Second run at 0d9468ac

The second run used the same machine and code page: `GetACP()` 1252, `GetOEMCP()` 850. The branch
was pulled to `0d9468ac` ("A decision model in CI, environment values as UTF-8 on Windows, speech
flags for the launcher") at llama.cpp b12283. The patch set has 20 patches, 0002 to 0030.

### Static test build and tests (measured)

The build used `.github\build.bat` with the step 1 flags in a fresh `llama\build`. The previous
directory was renamed to `llama\build.1e4e764a-bak`. The build was green, with this build-info line:

```
jllama 6.1.0-SNAPSHOT (0d9468ac0f999b313a3944d2f26952148dfd1d79), patches 0002,0003,0006,0007,0008,0014,0015,0017,0018,0020,0021,0022,0023,0024,0025,0026,0027,0028,0029,0030
```

`jllama_test.exe --gtest_filter=CApiNonAsciiPaths.*:CApiSlotFiles.*` gave `[  PASSED  ] 9 tests.`,
with exit code 0.

### Counter-check of the new sites (measured)

The method is the same as in step 3: the patch file is moved out, a fresh `llama\build-no00NN` is
configured, `--target jllama_test` is built, and the patch is moved back.

| Test | all 20 | without 0029 | without 0030 | without 0025 |
|---|---|---|---|---|
| CApiSlotFiles.ASlotFileIsNamedInsideTheSlotSaveDirectory | green | green | green | green |
| CApiSlotFiles.ASlotDirectoryWithNonAsciiCharactersHoldsTheWholeSlotFile | green | green | green | **red** |
| CApiSlotFiles.ASlotFileNameWithNonAsciiCharactersIsOneFileThatRestoresWhole | green | green | green | **red** |
| CApiSlotFiles.ASlidingWindowModelsSlotFileCarriesItsContextCheckpoints | green | green | green | **red** |
| CApiNonAsciiPaths.TheModelFilesAreReadUnderANonAsciiPath | green | green | green | (not run, then green)¹ |
| CApiNonAsciiPaths.TheMediaDirectoryIsCheckedAsUtf8 | green | **red** | green | (not run, then green)¹ |
| CApiNonAsciiPaths.AnOptionFromTheEnvironmentIsReadAsUtf8 | green | green | **red** | (not run, then green)¹ |
| CApiNonAsciiPaths.TheLogFileIsWrittenUnderItsNonAsciiName | green | green | green | (not run, then green)¹ |
| CApiNonAsciiPaths.TheImatrixQuantizerAndTrainerReadAndWriteUnderANonAsciiPath | green | green | green | (not run, then green)¹ |

**Result:** each of the three expected tests is red without its patch. None stays green.

Without 0025 the two older slot tests are red as well. This is expected: 0025 is their patch too,
for the draft appendix.

¹ Without 0025 the process ended after the third red slot test. That is the test defect described
below. With the defect fixed in the test, the five `CApiNonAsciiPaths` tests ran and were green.

The patch lists the configure step reported:

- without 0029: `…,0028,0030`
- without 0030: `…,0028,0029`
- without 0025: `…,0024,0026,…,0030`

After the three variants `llama\build` was re-configured: `jllama-build.txt` names all 20 patches
and `jllama.commit=0d9468ac…`, without `-dirty`.

#### Failure output, verbatim

**Without 0029:**

```
X:\...\llama\src\test\cpp\test_c_api.cpp(3913): error: Expected equality of these values:
  jllama_chat_completions(loaded, look_at("file://b\xC3\xA4r.png").c_str(), &seen.s)
    Which is: 2
  JLLAMA_STATUS_OK
    Which is: 0
file does not exist or cannot be opened: bär.png
```

**Without 0030:**

```
error while handling environment variable "LLAMA_ARG_SLOT_SAVE_PATH": No mapping for the Unicode character exists in 
the target multi-byte code page.
X:\...\llama\src\test\cpp\test_c_api.cpp(3963): error: Expected: (loaded) != (nullptr), actual: NULL vs (nullptr)
Failed to parse model parameters
```

**Without 0025.** In the directory shape with a draft model:

```
X:\...\llama\src\test\cpp\test_c_api.cpp(1744): error: Expected equality of these values:
  jllama_slot_action(model, JLLAMA_SLOT_SAVE, 0, name.c_str(), &saved.s)
    Which is: 1
  JLLAMA_STATUS_OK
    Which is: 0
Unable to save slot: incomplete draft state
```

In the name shape with a draft model:

```
I slot save_slot_dr: id  0 | task -1 | appended the draft state (0.015 MiB) to 
'C:\Users\<user>\AppData\Local\Temp\jllama-slots-ascii-3371304828\f-ä-日-draft.bin'
X:\...\llama\src\test\cpp\test_c_api.cpp(1813): error: Expected equality of these values:
  files.size()
    Which is: 2
  1u
    Which is: 1
the slot file and its appendix must be one file
```

In the sliding-window test:

```
X:\...\llama\src\test\cpp\test_c_api.cpp(2138): error: Expected equality of these values:
  files.size()
    Which is: 2
  1u
    Which is: 1
the slot file and its checkpoints must be one file
```

The log line in the name shape says "appended … to" the real name. The appendix stream, however,
opened the name as the active code page reads it, so it created a second file.

### Found: a failing sliding-window test ended the whole run (measured, fixed in the test)

Without 0025, `jllama_test.exe` ended after `CApiSlotFiles.ASlidingWindowModelsSlotFileCarriesItsContextCheckpoints`:

- the next test printed its `[ RUN ]` line, and nothing came after it;
- the exit code was **3**, not the 1 of a failed run;
- the five `CApiNonAsciiPaths` tests that follow went unreported.

Reproducible, and isolated by these runs:

- the same filter run again: exit code 3 at the same place;
- `CApiNonAsciiPaths.*` alone: 5/5 green, exit code 0;
- each of the three red slot tests alone, followed by `TheModelFilesAreReadUnderANonAsciiPath`:
  exit code 1, 1 and **3**. Only the sliding-window test leaves the process broken.

**Cause, read from the test.** The test installs `jllama_set_log_callback(…, &lines)`, where `lines`
is a local on the test's stack. It removes the callback only at its end, after
`jllama_model_free`. The `ASSERT_EQ(files.size(), 1u)` at line 2138 returns from the test function
before that. The callback stays installed with a pointer into the dead frame, and the first log
line of the next test locks a mutex in freed stack memory.

Only a failing run reaches this, which is why a green CI never shows it. When it does happen, it
hides every later test's result.

**Fix, in the test only.** A guard object directly after the callback is installed resets it in its
destructor (`struct LogCallbackReset { ~LogCallbackReset() { jllama_set_log_callback(nullptr, nullptr); } }`),
on every path out of the test. It is formatted with clang-format 23.1.5; the diff is the 6 added lines.

Measured after the fix:

- `llama\build` (all 20 patches): 9/9 green, exit code 0;
- `llama\build-no0025` (rebuilt with 0025 moved out again): the three slot tests red, the five
  `CApiNonAsciiPaths` tests run and are green, exit code 1.

### Java layer: `--log-file`, slots, quantizer under `C:\Temp\Prüfung\` (measured)

The probe ran with Temurin 25.0.5+8 against the static `jllama.dll` of this build, through the
module's runtime classpath, `target\classes` (recompiled at `0d9468ac`) and `src\main\natives`. The
paths are built in Java from `\u` escapes; step 4 gives the reason. The model is
`AMD-Llama-135m-code.Q2_K.gguf`, copied to `C:\Temp\Prüfung\日本\model.gguf`.

| Call | Path | Result |
|---|---|---|
| `setLogFile` | `C:\Temp\Prüfung\日本\jllama.log` | ok: 1911 bytes, the only new file in the directory |
| `setLogFile` | `C:\Temp\Prüfung\log-ä-日.log` | ok: 1911 bytes |
| `setSlotSavePath` + `saveSlot`/`restoreSlot` | dir `C:\Temp\Prüfung\slots-日`, file `slot.bin` | ok: `n_written` 553540 = `n_read`, one file of 553540 bytes |
| `setSlotSavePath` + `saveSlot`/`restoreSlot` | dir `C:\Temp\jllama-slots-ascii`, file `s-ä-日.bin` | ok: `n_written` 553540 = `n_read`, one file |
| `LlamaQuantizer.quantize` (Q8_0, requantize) | `…\日本\model.gguf` → `…\日本\q8-ä.gguf` | ok: 143274624 bytes, loads and answers |
| `computeImportanceMatrix` | text `…\日本\calib-ä.txt` → `…\日本\imatrix-ä.gguf` | ok: 6 chunks |
| `mergeImportanceMatrices` (legacy) | → `C:\Temp\Prüfung\imatrix-日.dat` | ok: 6 chunks |
| `LlamaQuantizer.quantize` (Q4_0) with the legacy matrix | → `C:\Temp\Prüfung\q4-日.gguf` | ok: 82665792 bytes |

**Observed with the log file.** The `--log-file` of a model stays attached to the process-wide log
after `close()`. In the probe, `log-ä-日.log` grew from 1911 to 7736 bytes while the later models of
the same JVM ran: the slot and quantizer cases, which set no log file.

That is the behaviour of `common_log_main()`, which is process-wide (read). The C API test
`TheLogFileIsWrittenUnderItsNonAsciiName` detaches it with `common_log_set_file(…, nullptr)`; Java
has no such call. It does not concern non-ASCII paths, but it means a Java caller's log file is
never closed while the JVM lives. On Windows that also blocks deleting it.

### Java layer: the extraction directory (measured)

| Setting | Extraction directory | Result |
|---|---|---|
| `-Djava.io.tmpdir=C:\Temp\Prüfung\tmp` | `C:\Temp\Prüfung\tmp\jllama-backend-cpu-…` | **ok**: extracted, `native backend 'cpu' loaded`, the model answers |
| `net.ladenthin.llama.tmpdir=C:\Temp\Prüfung\tmp-日` (set in code; the launcher reads `-D` in Cp1252) | `C:\Temp\Prüfung\tmp-日\jllama-backend-cpu-…` | **fails** |

The failing case, verbatim (the PATH list is cut):

```
RESULT FAILED load C:\Temp\Prüfung\日本\model.gguf -- java.lang.UnsatisfiedLinkError: The native library of backend 'cpu' could not be loaded for os.name=Windows, os.arch=x86_64 (GPU modules: none); reason: C:\Temp\Prüfung\tmp-?\jllama-backend-cpu-1a1c9b9f0e4-c4c200\jllama.dll: Can't find dependent libraries; paths=[…]
```

The files are there: `jllama.dll`, `jllama.lib` and `jllama_api.h` are under the real
`C:\Temp\Prüfung\tmp-日\…` directory, written by Java. The load is what fails. The JVM's message
shows the path with `?` in place of `日`, while `ü` is intact. So `System.load` hands the path to
Windows in the ANSI code page (Cp1252, `sun.jnu.encoding`), and a character with no representation
there becomes `?`. The path then names a directory that does not exist.

That reading comes from the message and is not verified in the JDK source. It is a limit of the JVM
(Temurin 25.0.5), not of the native library.

Consequence: an extraction directory with a character outside the ANSI code page cannot be loaded
from. That is a user profile, `TEMP` or `net.ladenthin.llama.tmpdir` with such a character, for
example a CJK user name on a 1252 system.

Within the code page it works: `ü` on 1252, as measured. On a CJK system the code page holds the
CJK characters, so the user's own profile name is normally representable (read, not measured).

Possible mitigations, not implemented, all read:

- the loader detects a directory `sun.jnu.encoding` cannot encode and falls back to another one;
- the loader loads through the directory's 8.3 short name, if one exists;
- at the least, the loader names this cause in its error message instead of "Can't find dependent
  libraries".

### Tests added to the branch

At the request to add a test where possible. Test code only; no library code was changed.

- **`llama/src/test/java/net/ladenthin/llama/NonAsciiPathIntegrationTest.java`** (new), 3 tests
  with the CI draft model (`TestConstants.DRAFT_MODEL_PATH`, skipped without it). Each uses a
  directory `d-ä-日` and file names `…-ä-日…`:
  - `aModelIsLoadedFromANonAsciiPath`;
  - `aSlotIsSavedAndRestoredUnderANonAsciiPath`: both shapes; asserts exactly one file, its size
    equal to `n_written`, `n_read` equal to `n_written` and `n_restored` equal to `n_saved`;
  - `theQuantizerAndTheImportanceMatrixWorkUnderANonAsciiPath`: requantization to Q4_0, then load
    and answer; an importance matrix from a calibration text; the legacy merge; a quantization
    reading the legacy file.

  It runs on every platform. On Linux and macOS it passes trivially; on the Windows Java jobs
  (code page 1252 on the runners) it is the measurement through JNI. Locally, with Temurin 25 on
  1252: `Tests run: 3, Failures: 0, Errors: 0, Skipped: 0`. Spotless (palantir-java-format) is clean.

  `--log-file` and the extraction directory are deliberately not in it:
  - the log file stays attached to the process after `close()`, so the `@TempDir` could not be
    removed on Windows;
  - the extraction directory is chosen once per JVM, before any test runs.
- **`llama/src/test/cpp/test_c_api.cpp`**: the `LogCallbackReset` guard in
  `ASlidingWindowModelsSlotFileCarriesItsContextCheckpoints`, described above. +6 lines.

## Left on disk (not in git)

- `llama\build`: `0d9468ac`, all 20 patches.
- `llama\build-no0029`, `llama\build-no0030`, `llama\build-no0025`: the counter-check builds of the
  second run.
- `llama\build-no0026`, `llama\build-no0027`, `llama\build-no0028`: the counter-check builds of the
  first run.
- `llama\build.1e4e764a-bak`: the `llama\build` of the first run.
- `llama\build.b12242-bak`: the `llama\build` before the first run.
- `llama\build-commit`, `llama\build-fix`, `llama\build-fix2`, `llama\build.b11639-bak`: from the
  earlier report.
- `llama\src\main\natives\...\Windows\x86_64\cpu\`: `jllama.dll` from the step 1 build, and the
  provenance files restored to it. Git ignores the directory.
- In the scratch directory, outside the repository: Temurin 25.0.5+8 and the probe.
- `C:\Temp\Prüfung\`: everything was deleted after each run.
