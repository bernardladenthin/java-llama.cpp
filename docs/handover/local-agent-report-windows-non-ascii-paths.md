<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT
-->

# Local agent report -- Windows, non-ASCII paths (llama.cpp b12242)

Answers the handover prompt "measure and report, nothing committed, nothing pushed", on branch
`claude/wonderful-tesla-1fkezz` at **`3f63d8be`** (llama.cpp **b12242**, `3b4d6ab3`). Every line marked
**measured** was run on this machine on 2026-11-10. Every line marked **read** comes from the source
and was not run. Where a step could not be done as asked, the report says so.

## Summary

| Question | Answer |
|---|---|
| Where "not a directory" comes from | `common/arg.cpp:3647` (upstream line; `:3652` in the patched tree) calls `fs_is_directory()`, `common/common.cpp:944`, which builds `std::filesystem::path` **from the narrow string**. MSVC decodes it in the ANSI code page (1252 here) and the UTF-8 directory does not exist under that name. **Measured** and **read** |
| Is the check the only obstacle for slots? | **No.** With the check fixed (one line), a slot without a draft model round-trips. With `--spec-type draft-simple` the **save fails** ("Unable to save slot: incomplete draft state"), because the appendix goes through `std::ofstream` over the same narrow string. A 9348-byte main-state file stays behind, and a restore of it "works" without the draft state. **Measured** |
| A silent split | **Present today, without any patch**: a non-ASCII *file name* in an ASCII slot directory (`saveSlot(0, "f-ä-日-draft.bin")`) writes the slot file under its real name and the draft appendix into a **second file with a mis-decoded name** (`f-Ã¤-æ—¥-draft.bin`, 9216 bytes). The server reports success with `n_written` 18564, which is both files together; the restore reads 9348 bytes and reports success. TODO.md says this "cannot occur through this option"; that is wrong for the file-name half. **Measured** |
| Other arguments | `-m`, `--mmproj`, `--lora`, `--shard`, the quantizer's input and imatrix paths, `jllama_imatrix`'s GGUF output and all of `jllama_finetune`'s model/output paths **work**. `--log-file`, the quantizer's **output**, the legacy imatrix output, `jllama_imatrix`'s text input, `jllama_finetune`'s `training_file` and **every download path** (`LLAMA_CACHE`, `--model-url` with `-m`) are **refused or write to a mis-decoded name**. The full table is below. **Measured** |
| UTF-8 code page | In a process whose code page is UTF-8 (an `activeCodePage` manifest on a copy of `jllama_test.exe`, no reboot), **every case above works**, the slot round trip with a draft model included. **Measured** |
| A side finding that blocks CI | `3f63d8be` **does not compile on Windows**. `test_c_api.cpp` includes `<windows.h>` without `NOMINMAX` (since `cd85dc98`), so `std::min` at line 1327 becomes the `min` macro. Measured locally with MSVC 19.44; in Publish run 42798707274 the jobs "C++ Tests Windows 2025 x86_64 (clang)", "C++ Tests Windows 11 arm64 (clang-cl)" and "Build and Test Windows 2025 x86_64 (MSVC)" fail on that line. **Fixed on the branch by `65bdaf45`** |
| Slots fully fixed by | **0024 + 0025: 2 files, 5 one-line sites** (`fs_is_directory`, plus the four appendix streams in `server-context.cpp`). **Measured**: ctest 2/2 and all six slot probe cases round-trip with a draft model |
| Recommendation | **Patch, in two layers**: llama.cpp patches for the upstream sites (0024 + 0025 for slots; downloads and `--log-file` are separate). The fixes for the jllama-owned sites (3 `std::fstream` sites) go in-tree. Keep the README limitation until the download half is done. The details are at the end |

## Environment

| | |
|---|---|
| OS | Windows 11 Pro 10.0.26300, x86-64 |
| Code pages | `GetACP()` = **1252**, `GetOEMCP()` = **850**, `GetConsoleCP()` / `GetConsoleOutputCP()` = 850 inside the test process (**measured**, printed by the probe). `chcp` in a fresh `cmd`: **850**; the agent's own PowerShell console was switched to 65001 by its host, which affects nothing measured here |
| "Beta: Unicode UTF-8 für weltweite Sprachunterstützung" | **Off.** `HKLM\SYSTEM\CurrentControlSet\Control\Nls\CodePage`: `ACP=1252`, `OEMCP=850`. With the beta on, both read 65001 |
| Compiler / generator | MSVC 19.44.35229 (VS 2022 Community, `vcvars64.bat`), Ninja Multi-Config (1.11.1 from VS), CMake 3.31.6-msvc6 |
| Java | **Temurin 25 is not installed.** Only Oracle JDK 21.0.9 is on the machine, and `mvn -q -f llama/pom.xml compile` **fails** under it: NullAway in JSpecify mode needs JDK 22+ or a JDK 21 that accepts `-XDaddTypeAnnotationsToSymbol=true`, and Oracle's does not (same finding as `local-agent-report-b11538-windows.md`). The native build does not need the Java classes, so every measurement below goes through `jllama_test.exe` (the C API), which the prompt allowed for step 4. The Java layer itself is **not** measured |
| Downloads | a local `python -m http.server` on 127.0.0.1 serving `models/stories260K.gguf` |
| Probe directories | ASCII root under the session scratch dir; every case writes below its own ASCII case directory, so a mis-decoded sibling shows up in the listing |

### Step 1 -- build and baseline

- `.github\build.bat -G "Ninja Multi-Config" -DOS_NAME=Windows -DOS_ARCH=x86_64 -DBUILD_TESTING=ON`:
  - **The configure succeeds; the build fails** on `test_c_api.cpp(1327): error C2589` (`std::min` vs the `min` macro, see the summary). For all measurements I added `#define NOMINMAX` before `#include <windows.h>` locally. After that the build is green.
  - The old `llama\build` (b11639) was moved aside to `llama\build.b11639-bak` and not deleted. The patch stamp would have refused to reconfigure it at b12242.
- **Measured:** `ctest --test-dir llama\build -C Release --no-tests=error --output-on-failure -R CApiSlotFiles` gives **2/2 passed**.
  - `ASlotDirectoryWithNonAsciiCharactersHoldsTheWholeSlotFile` passes because it takes its `GetACP() != CP_UTF8` branch (1252), which requires the load to fail.
  - The log line is `error while handling argument "--slot-save-path": not a directory: ...\d-ä-日`.
  - `jllama_last_error()` says only `Failed to parse model parameters`; the reason is only in the log.

## Step 2 -- the line that says "not a directory"

**Read** (b12242, upstream line numbers; the fetched tree is shifted by our patches to `arg.cpp:3652`):

```cpp
// common/arg.cpp:3642-3655
add_opt(common_arg(
    {"--slot-save-path"}, "PATH",
    "path to save slot kv cache (default: disabled)",
    [](common_params & params, const std::string & value) {
        params.slot_save_path = value;
        if (!fs_is_directory(params.slot_save_path)) {
            throw std::invalid_argument("not a directory: " + value);   // <- the message
        }
        ...

// common/common.cpp:944-947
bool fs_is_directory(const std::string & path) {
    std::filesystem::path dir(path);    // narrow string -> path: MSVC decodes it in the ANSI code page
    return std::filesystem::exists(dir) && std::filesystem::is_directory(dir);
}
```

- **The API.** The check is `std::filesystem::path` constructed from a narrow `std::string`. It does not use `_stat` or a `fs_*` helper that converts.
  - On MSVC that constructor converts with the process's active code page.
  - The UTF-8 bytes `c3 a4` (ä) become `Ã¤`, and `e6 97 a5` (日) becomes `æ—¥`, so `exists()` looks for a directory that is not there.
- **The same helper, the same defect.** `--media-path` (`common/arg.cpp:3661`) uses the same helper. These two are its only callers.

## Step 3 -- with the check fixed

- **The fix.** I changed only `fs_is_directory` to read the string as UTF-8. It went in as a local patch file, `llama/patches/0024-common-fs-is-directory-reads-utf8.patch`, so the real applier applied it; the configure log reports `patches 0002,...,0023,0024`.
- **The test.** I removed the `GetACP() != CP_UTF8` branch from `ASlotDirectoryWithNonAsciiCharactersHoldsTheWholeSlotFile`, so it now expects the load to succeed.
- **The build.** I built into a fresh `llama\build-fix`. A hand edit in `_deps\llama.cpp-src` is rejected by the applier's tree fingerprint ("has local modifications that do not match the current patch set"), which is the guard working as designed.

**Measured**, `ctest --test-dir llama\build-fix -C Release -R CApiSlotFiles`: **1 of 2 fails**.

- **a) Without a draft model: works.**
  - **Load:** ok.
  - **Save:** `n_written` 9348 = 9348 bytes on disk, under the real name.
  - **Restore:** `n_restored` 34 = `n_saved` 34, `n_read` 9348.
- **b) With `--spec-draft-model <same model> --spec-type draft-simple`: the save fails, loudly.**
  - `test_c_api.cpp:1741`: `jllama_slot_action(SAVE)` returns `JLLAMA_STATUS_ERROR`, `Unable to save slot: incomplete draft state`.
  - The server logs `failed to append the draft state to '...\jllama-slots-ä-日本-...\draft.bin'`.
  - `std::ofstream(filepath, binary|app)` (`tools/server/server-context.cpp`, `save_slot_draft`) opens `...\jllama-slots-Ã¤-æ—¥æœ¬-...\draft.bin`. That directory does not exist, so the stream fails, and **no file with a mis-decoded name is created**.
  - What does remain is the 9348-byte main state that `ggml_fopen` wrote before the appendix. A later restore of that name succeeds with `n_read` 9348 and no draft state; the draft context starts empty.

The probe repeats this with listings after load, save and restore. Names are shown as UTF-8 bytes, read through the wide API; `dir /x` gives the 8.3 names.

```
--slot-save-path dir:ä+日 +draft      (patched, build-fix)
after load     d-ä-日\                [64 2d c3 a4 2d e6 97 a5]   <DIR>   (8.3: D--~1)
after save     d-ä-日\f-draft.bin     [66 2d 64 72 61 66 74 2e 62 69 6e]   9348 bytes
               -> save status=error "Unable to save slot: incomplete draft state"
after restore  d-ä-日\f-draft.bin     9348 bytes  -> restore ok, n_read=9348 (no draft state)

--slot-save-path name:ä+日 +draft     (UNPATCHED build and patched build alike: the check is not involved)
after save     f-ä-日-draft.bin       [66 2d c3 a4 2d e6 97 a5 2d 64 72 61 66 74 2e 62 69 6e]   9348 bytes   (F---DR~1.BIN)
               f-Ã¤-æ—¥-draft.bin    [66 2d c3 83 c2 a4 2d c3 a6 e2 80 94 c2 a5 2d 64 72 61 66 74 2e 62 69 6e]   9216 bytes   (F---DR~2.BIN)
               -> save status=ok, n_written=18564 (= 9348 + 9216), on disk under the real name: 9348
after restore  -> status=ok, n_restored=34, n_read=9348: the draft state is silently gone
```

- **The silent second file exists, and it exists today.** The file name passes `fs_validate_filename` and jllama's own check. The main state goes through `ggml_fopen` (UTF-8, correct name). The appendix goes through `std::ofstream` (ANSI, mis-decoded name in the same ASCII directory, which does exist). The response's `n_written` counts both files, so the caller cannot tell.
- **The context-checkpoint appendix (read).** It (`save_slot_checkpoints` / `load_slot_checkpoints`) uses the same `std::ofstream` / `std::ifstream` pattern. The tiny llama model makes no checkpoints, so it is **not measured**; it will behave exactly like the draft appendix.
- **Does fixing the appendix too give the full round trip?** Yes, **measured** in "Step 3b" below.

## Step 4 -- the other path arguments

**Measured** with `jllama_test.exe` (C API) on the unpatched build. The results on the 0024 build are identical except for the `--slot-save-path` directory rows.

The shapes:

- **dir** = a directory `d-ä-日` with an ASCII file name;
- **name** = an ASCII directory with the file name `f-ä-日`;
- an all-ASCII control ran green for every row and is not repeated here.

"Works" means the call succeeded, the file exists under its real name with the expected size, and nothing else appeared below the case directory.

| Argument | API reaching the file (read) | dir: ä+日 | name: ä+日 |
|---|---|---|---|
| `-m` | `ggml_fopen` (UTF-8 to `_wfopen`) | works | works |
| `--mmproj` | `gguf` + `open_ifstream_binary` (UTF-8 to wide) | works | works |
| `--lora` | `ggml_fopen` | works | works |
| `--shard` | `ggml_fopen` | works | works |
| `--log-file` | `fopen(path, "w")` narrow, `common/log.cpp:408` | **refused, silently**: load ok, no file anywhere | **wrong place**: `f-Ã¤-æ—¥.log`, 2297 bytes |
| `jllama_quantize_imatrix` output | `std::ofstream(fname)` narrow, `src/llama-quant.cpp:1260` | **refused**: "Quantization of '<input>' failed with code 1" (names the *input*) | **wrong place**: `f-Ã¤-æ—¥.gguf`, 518272 bytes |
| `jllama_quantize_imatrix` input path | `ggml_fopen` | works | works |
| `jllama_quantize_imatrix` imatrix path | `gguf_init_from_file` | works | works |
| `jllama_imatrix` output (GGUF) | `gguf_write_to_file` | works | works |
| `jllama_imatrix` text input | `std::ifstream(text_path)` narrow, `imatrix_engine.cpp:278` (**jllama's own**) | **refused**: "could not read the calibration text" | **refused**: same |
| `jllama_imatrix_merge` output, legacy format | `std::ofstream(output_path)` narrow, `imatrix_engine.cpp:229` (**jllama's own**) | **refused**: "could not write the importance matrix" | **wrong place**: `f-Ã¤-æ—¥.dat`, 8463 bytes |
| `jllama_finetune` output | `llama_model_save_to_file` (`ggml_fopen`) | works | works |
| `jllama_finetune` model path | `ggml_fopen` | works | works |
| `jllama_finetune` `training_file` | `std::ifstream(cfg.training_file)` narrow, `train_engine.cpp:34` (**jllama's own**) | **refused**: "cannot open training file" | **refused**: same |
| `--slot-save-path` (directory) | `fs_is_directory`, see step 2 | **refused**: "not a directory" | n/a |
| slot file name in an ASCII directory | `ggml_fopen` + `std::ofstream` appendix | n/a | **split**: see step 3; silent with a draft model, works without |
| `--model-url` + `-m` target (download) | `std::ofstream` / `std::rename` / `std::filesystem::path(string)`, `common/download.cpp` | **wrong place, then load refused**: the download creates a sibling directory `d-Ã¤-æ—¥\f.gguf` (1185376 bytes) and the load of the real path fails | **wrong place, then load refused**: `f-Ã¤-æ—¥.gguf` |
| `LLAMA_CACHE` + `--model-url` (no `-m`) | `std::getenv` (narrow, ANSI) in `fs_get_cache_directory`, `common/common.cpp:981`; then `.string()` (ANSI) | `c-ä` (in cp1252): **downloaded to the right directory, load refused**. `c-日`, `c-ä-日`: **refused**, "failed to create cache directory: ...\c-?" | n/a |

How the download and `LLAMA_CACHE` rows were measured:

- `jllama_model_load` itself does **not** download: it parses with `LLAMA_EXAMPLE_SERVER`, and upstream skips the download there ("server will call common_params_handle_models() later", `common/arg.cpp`).
- With `--model-url` and no `-m`, even the all-ASCII control fails ("loading model ''"). I measured that directly; whether that is intended for `LlamaModel(setModelUrl)` is a question for the Java layer, which I did not measure.
- So the two download rows call the functions the embedded server calls, `common_models_handler_init` + `common_models_handler_apply` (`server.cpp`, `server_run_instance`), and then load the resulting `params.model.path` with `-m`. That is **measured through the same upstream functions, not through `jllama_server_start`**.

**Measured**, the narrow `getenv("LLAMA_CACHE")` bytes after setting the variable through the wide API, as a parent process would:

- `ä` arrives as `e4` (cp1252);
- `日` arrives as `3f` (`?`).

The ä case downloads correctly because `download.cpp` is consistently narrow. The load then fails because the ANSI string `...c-\xe4\stories260K.gguf` reaches `ggml_fopen`, which expects UTF-8.

**Not measured:**

- `--media-path`: same helper as `--slot-save-path`, read only;
- `--models-dir` in router mode, `-hf` downloads, the MCP configuration file and the server's file tools: TODO.md lists them, and they need the embedded server in router mode with child processes, which this session did not set up;
- the Java layer.

## Step 5 -- the same with a UTF-8 code page

- **The system-wide switch was not used:** the "Beta: UTF-8" setting needs a reboot. I used the process-level alternative instead.
- **The setup:**
  - a copy, `llama\build\Release\jllama_test_utf8.exe` (of the unpatched build), got an `activeCodePage=UTF-8` manifest merged in with `mt.exe`;
  - the manifest has `activeCodePage` only, not upstream's `longPathAware`, so that only one thing changes;
  - the test process then reports `GetACP=65001 GetOEMCP=65001`.
- **The caveat:** ctest is registered for `jllama_test.exe`, so the copy was run directly with `--gtest_filter=CApiSlotFiles.*` instead of `ctest -R CApiSlotFiles`.

**Measured:**

- `CApiSlotFiles.*`: **2/2 passed**. The non-ASCII test now takes its round-trip branch: draft model, appendix in the same file, `n_written` equal to the size on disk, restore equal.
- Every row of the step 4 table and every row of the slot probe: **works**. That includes `--log-file`, the quantizer's output, the legacy imatrix, `training_file`, the text input, all three `LLAMA_CACHE` shapes (`ä`, `日`, both) and both `-m` download targets. The slot probe also gives `n_written` 18564 = 18564 on disk with a draft model, in both the dir shape and the name shape.

So everything that fails above fails because of the code page, and nothing else. It is also proof that `java.exe` cannot get there on its own: a library cannot carry a manifest, and the JVM's executable does not have this one.

## Step 3b -- the appendix as well (0024 + 0025)

This went beyond the prompt, to back up the recommendation.

- **The patch.** `llama/patches/0025-server-slot-appendix-reads-utf8.patch` (local, not committed) changes the four appendix streams in `tools/server/server-context.cpp` from `std::ofstream ofs(filepath, ...)` / `std::ifstream ifs(filepath, ...)` to `...(std::filesystem::u8path(filepath), ...)`. Each is a one-line change; `<filesystem>` is already included.
- **The build.** A fresh `llama\build-fix2`; the applier reports `patches ...,0023,0024,0025`.

**Measured:**

- `ctest --test-dir llama\build-fix2 -C Release -R CApiSlotFiles`: **2/2 passed**. That is the flipped test ("the load succeeds"), with the draft model, the appendix in the same file, `n_written` equal to the size on disk, and a matching restore.
- The slot probe, all six cases (ascii, dir and name, each with and without a draft model), **work**:

```
--slot-save-path dir:ä+日 +draft    after save   d-ä-日\f-draft.bin   [66 2d 64 72 61 66 74 2e 62 69 6e]   18564 bytes
                                    save n_written=18564, restore n_read=18564, n_restored=34
--slot-save-path name:ä+日 +draft   after save   f-ä-日-draft.bin     [66 2d c3 a4 2d e6 97 a5 2d 64 72 61 66 74 2e 62 69 6e]   18564 bytes
                                    no second file; restore n_read=18564
```

So for slot files **0024 + 0025 are sufficient**: 2 files, 5 one-line sites. The checkpoint appendix shares the 0025 sites. It is not exercised by the tiny model, so that part is read, not measured.

## The smallest diff that fixes the slot check

```diff
--- a/common/common.cpp
+++ b/common/common.cpp
@@ -942,7 +942,7 @@ void fs_write_atomic(const std::filesystem::path & path, const std::string & dat
 }
 
 bool fs_is_directory(const std::string & path) {
-    std::filesystem::path dir(path);
+    std::filesystem::path dir = std::filesystem::u8path(path);
     return std::filesystem::exists(dir) && std::filesystem::is_directory(dir);
 }
```

- It applies cleanly through `apply-llama-patches.cmake` after 0002-0023. It is identical on every platform: `u8path` is the identity where the narrow encoding is already UTF-8. It also fixes `--media-path`.
- **Measured: it is not enough on its own.** It turns a loud refusal at load into a loud failure at save, and only with a draft model (or a model with context checkpoints). A slot without a draft model works with it.

## Recommendation

**Patch, do not leave it as documented. But the slot check alone is the wrong unit: it moves the failure from load time to save time.**

1. **Slot files: a llama.cpp patch, 2 files, 5 sites.**
   - `common/common.cpp` (1 site, `fs_is_directory`, 0024 above).
   - `tools/server/server-context.cpp` (4 sites: `save_slot_checkpoints`, `load_slot_checkpoints`, `save_slot_draft` and `load_slot_draft`, each opening `std::filesystem::u8path(filepath)` instead of `filepath`; 0025).
   - Those 4 sites also fix the **silent split for a non-ASCII file name**, which exists today without any patch and is the worse half: data is lost and success is reported.
   - **Measured in step 3b:** 0024 + 0025 together give the full round trip, with and without a draft model, for both shapes.
   - The flipped `ASlotDirectoryWithNonAsciiCharactersHoldsTheWholeSlotFile` is the test for it. Add a name-shaped case (an ASCII directory with a file name `ä日`) alongside it, because only that case shows the split.
2. **jllama's own sites: fix them in-tree, not as a patch, 2 files, 3 sites.** Open the stream on `std::filesystem::u8path(path)`:
   - `imatrix_engine.cpp:229` (legacy output) and `:278` (calibration text);
   - `train_engine.cpp:34` (`training_file`).
3. **`--log-file`: a llama.cpp patch, 1 site.** `common/log.cpp:408`, `fopen` becomes `ggml_fopen`. `log.cpp` does not include `ggml.h` today, so the patch would add that include.
4. **Quantizer output: a patch in `src/`, 1 site.** `src/llama-quant.cpp:1260`, `std::ofstream(fname)`. It is a core-library file, not `common/`, so it is the least likely to be taken upstream as-is.
5. **Downloads and `LLAMA_CACHE`: the expensive part.**
   - `common/download.cpp`: the `.etag` read/write, the partial file `std::ofstream`, `std::rename`, `create_directories(path(string))` and the `exists()` checks.
   - `common/common.cpp`: `getenv("LLAMA_CACHE")` would become `_wgetenv`, and `fs_get_cache_file(...).string()` would become `u8string()` in `get_default_local_path`, `common/arg.cpp:331`.
   - `hf-cache.cpp` probably too. A rough grep count of narrow file-API calls is 12 in `download.cpp`, 11 in `common.cpp` (not all of them path sites), 1 in `hf-cache.cpp` and 3 in `preset.cpp`. Each needs reading.
   - That matches TODO.md's "about 30 lines over `common/{common,download,hf-cache,preset}.cpp` and `tools/server/server-{context,mcp,models,tools}.cpp`". It is **read, not measured**, beyond the two download rows above.
   - Recommendation: a separate patch, refreshed at each bump. Or, if it is judged too large, keep it documented ("`LLAMA_CACHE` and download targets must be ASCII on Windows unless the code page is UTF-8"). Either way, report it upstream as the embedding-host half of the app-manifest change: upstream's own executables are UTF-8 through `common/win-app.manifest`, and a library inside a host cannot be.
6. **Until 1-4 land,** keep README's slot-directory sentence, and extend it:
   - the slot *file name* must be ASCII on Windows too, or a draft/checkpoint appendix is lost silently;
   - the same applies to `--log-file`, the quantizer output, the legacy imatrix output and the calibration/training text files.
   - Correct TODO.md's "the silent second file expected above cannot occur through this option".
7. **Independently of all of this, the CI fix:** `#define NOMINMAX` before `#include <windows.h>` in `test_c_api.cpp`. Without it, `3f63d8be` does not build on any Windows C++ job. **Done in `65bdaf45`.**

`setlocale(LC_ALL, ".UTF-8")` would make every narrow site UTF-8 at once and would avoid patches 1-5. It changes the C locale of the whole JVM process, though: the UCRT is shared, which TODO.md already notes. It was not measured here.

## What the commit carries, and what stayed in this report

**In the commit:**

- `llama/patches/0024-common-fs-is-directory-reads-utf8.patch` and
  `llama/patches/0025-server-slot-appendix-reads-utf8.patch`, each with a header that names the defect,
  the b12242 file:line and the measurement.
- `test_c_api.cpp`:
  - `CApiSlotFiles.ASlotDirectoryWithNonAsciiCharactersHoldsTheWholeSlotFile` no longer pins the
    refusal on a non-UTF-8 code page. It requires load, save and restore on every code page, without
    and with a draft model, and additionally that the restore's `n_read` equals the save's
    `n_written`.
  - **New** `CApiSlotFiles.ASlotFileNameWithNonAsciiCharactersIsOneFileThatRestoresWhole`: an ASCII
    directory and the file name `f-ä-日-{plain,draft}.bin`; exactly one file in the directory, its
    size equal to `n_written`, the restore's `n_read` equal to `n_written`.
  - **New** `CApiNonAsciiPaths.TheModelFilesAreReadUnderANonAsciiPath` (`-m`, `--shard`, `--mmproj`,
    `--lora`) and `CApiNonAsciiPaths.TheImatrixQuantizerAndTrainerReadAndWriteUnderANonAsciiPath` (the
    model input of `jllama_imatrix`, `jllama_quantize_imatrix` and `jllama_finetune`; the GGUF output
    of `jllama_imatrix` and the imatrix input of the quantizer; the trainer's output). Each runs in both
    shapes, a non-ASCII directory and a non-ASCII file name. These are the probe rows that work without
    any patch, kept as a guard against a regression to a narrow stream.

**Left out of the temporary `NonAsciiProbe` suite**, with the results kept in the tables above:

| Probe case | Why it stayed out |
|---|---|
| `Environment` | prints the code pages only, a measurement |
| `LogFile` (`--log-file`) | red on code page 1252 without a further patch (`common/log.cpp:408`, narrow `fopen`) |
| `QuantizeOutputAndImatrixInput`, the output path | red without a further patch (`src/llama-quant.cpp:1260`). The input and imatrix rows went into `CApiNonAsciiPaths` |
| `ImatrixOutputAndTextInput`, the text input and the legacy `jllama_imatrix_merge` output | red without a fix of jllama's own `imatrix_engine.cpp:278` / `:229`. The GGUF output row went into `CApiNonAsciiPaths` |
| `Finetune`, `training_file` | red without a fix of `train_engine.cpp:34`. The model and output rows went into `CApiNonAsciiPaths` |
| `LlamaCacheWithModelUrl` | a pure measurement: `jllama_model_load` does not download at all (even the ASCII control fails), and it needs a local HTTP server |
| `DownloadThroughTheServersModelHandler` | red for every non-ASCII shape without a download patch, and it needs a local HTTP server |
| `SlotSaveRestore` | covered by the two `CApiSlotFiles` tests, which assert what the probe only printed |

**Not committed (local only):**

- the build dirs `llama/build`, `llama/build-fix`, `llama/build-fix2` and `llama/build-commit`;
- `llama/build.b11639-bak`, the earlier b11639 build dir, moved aside;
- `jllama_test_utf8.exe`, the manifest copy for step 5.

`docs/patches.md` is not regenerated in this commit, so `check-patches.py` reports it as stale until it
is rendered.
