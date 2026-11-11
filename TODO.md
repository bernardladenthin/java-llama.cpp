<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT
-->

# TODO — java-llama.cpp

Open work items and the decisions that constrain them. Cross-cutting tracking lives in
[`../workspace/crossrepostatus.md`](../workspace/crossrepostatus.md);
items here are jllama-specific or are this repo's slice of a
cross-cutting initiative.

**Completed work is not recorded here.** It lives in git history and in
`crossrepostatus.md`; a finished item is deleted from this file rather than annotated,
so entries below describe remaining work or an explicit deferral policy.

## After the 6.0.0 release — open

6.0.0 is published (Central, GitHub Release, attestations; Publish run 40239731871); `main` is at
`6.1.0-SNAPSHOT`. What the release left open:

- [ ] **The snapshot path of bernardladenthin/actions, read in dispatch
      [40227501358](https://github.com/bernardladenthin/java-llama.cpp/actions/runs/40227501358)**
      (`081457f0`, b11887, actions v1.0.1, `publish_to_central`): every build, test and smoke job
      green, and so is the gate `Prepare and verify signed snapshot artifacts` -- the
      `_remote.repositories` fix of v1.0.1 holds, the Android AARs' timestamped Gradle names map
      back -- and `attest-snapshot`. Two consumers failed, neither on the release path:
      - `github-snapshot`: uploaded the assets, then `PATCH git/refs/tags/snapshot` with the workflow
        token was refused (`Resource not accessible by integration`, 403), so the `snapshot` tag and
        notes still name `53f307ac` (May) and nothing was pruned. Presumably the workflow token may
        not move a ref across commits that change `.github/workflows/` (12 files between the two);
        not verified. actions v1.0.3 (and later; v1.1.0 is pinned) deletes and re-creates the tag on that 403 and points
        the release at it; whether GitHub accepts that shows only a snapshot dispatch that gets that far:
        [40367798517](https://github.com/bernardladenthin/java-llama.cpp/actions/runs/40367798517)
        (`efc279c8`, 2026-10-26) and
        [40699518595](https://github.com/bernardladenthin/java-llama.cpp/actions/runs/40699518595)
        (`c8de6e7a`, 2026-10-28) both failed on cancelled macOS arm64 build jobs (runner capacity), so
        the snapshot jobs were skipped and the tag is still `53f307ac`. Done when a dispatch's
        `github-snapshot` has moved the `snapshot` tag to its commit and `central-snapshot` has passed
        `superseded()` without a 502.
        `github-release` does not move a tag (the owner's `v6.0.0` exists already).
      - `central-snapshot`: `superseded()` died on a GitHub API `502` listing the jobs of an older
        run, in the re-run too: with 100 per page GitHub fails that listing for large runs, with 50
        it answered every try. actions v1.0.3 (and later) lists 50 per page and repeats GitHub reads on
        5xx. `central-release` does not call it.
- [ ] **Owner:** turn off GitHub's "Automatic dependency submission" (repository settings, Code
      security). The generated run fails on every push (it has no JDK 22+ toolchain);
      `dependency-submission.yml` (JDK 25, `push` on `main`) replaces it. A bot token cannot change
      the setting (403).
- [ ] **Owner: the open points of the natives license texts** (`llama/legal/natives/`, in every
      natives jar since 2026-10-25): whether the oneMKL SYCL headers compiled into the two SYCL
      modules need Intel's Simplified Software License text (it ships only in Intel's installer); the
      provenance of the DRY sampler in `src/llama-sampler.cpp` (upstream treats it as an MIT
      contribution by its author; KoboldCpp, where it came from, is AGPL-3.0); the third-party logos
      the WebUI embeds (trademarks, no license text). The WebUI package list is an upper bound
      (imports + lockfile); an exact list from the bundle would need a plugin in upstream's build.
- [ ] After publication: carry out the upstream submissions queued below (`main` is at
      `6.1.0-SNAPSHOT` since the post-release bump).
- [ ] **The automated post-release check** (`.github/workflows/post-release.yml`,
      docs/post-release-check.md "Automated") is proven by hand: the `workflow_dispatch` with `v6.0.0`
      (run 40515663015, 2026-10-27) passed the 8 audit checks, `post-release-check.py` (failing android
      on exactly the expected checker-qual finding, fixed since) and all four Java 8 consumers, and
      `LocalCheck` / `FfmCheck` passed against the 6.0.0 jars from Central. Open: the first
      `workflow_run` after 6.1.0. The `workflow_run` runs after #573 and since (40367736958,
      40367890447, ...) ended `skipped`, correctly: no `v*` tag is behind those Publish runs.

## Delivery roadmap — remaining work

The remaining items retain the numbering of the 2026-10-15 roadmap referenced by `CLAUDE.md`.
Done and therefore deleted here, while comments in the code still name them by number: **1** the
provenance record (`jllama-build.txt` keys, `jllama-sbom.json`, `docs/patches.md`; CLAUDE.md,
"Modular natives"), **2** the natives directory as a tested contract (`lib.path`; CLAUDE.md, "Native
Library Resolution"), **4** export hygiene (the export lists; CLAUDE.md, "Natives jars"), **6** the
signed source archive of a release (CLAUDE.md, "Source archive"; the push of the pinned commit to a
fork was removed again on 2026-10-25, the owner maintains that fork), **7** layer B, the `jllama_*`
C API (CLAUDE.md, "The C API"; `docs/c-api.md`), **9** the decision for 6.0.0
(6.0.0 is released; CHANGELOG) and **10** `llama-ffm` (CLAUDE.md, "Java 22+ without JNI").
Windows code signing stays deferred by the owner's decision. Vendor hash pinning and the
measurements below remain open hardening/coverage work; they do not establish a runtime defect.

### 3. Hard pinning (C.2) — four repos

- [ ] **Pin the vendor checksums into `publish.yml`** (`OPENVINO_LINUX_SHA256`,
      `OPENVINO_WINDOWS_SHA256`, `VULKAN_SDK_SHA256`). The fallback verifies downloads against
      vendor-hosted checksums, but an archive and checksum replaced together would still pass.
      Obtain the values from a trusted download or the successful Publish logs and verify them
      before pinning. `CUDA_REDIST_MANIFEST` is already version-pinned.
- [ ] **Hash the oneAPI installers and Level Zero downloads.** Both SYCL jobs use versioned
      `ONEAPI_DLE_URL` / `ONEAPI_ONEDNN_URL` installers (Deep Learning Essentials 2026.2.0 and
      oneDNN 2026.1.0 since b12283). Download them on a machine that can reach Intel, then add pinned SHA-256
      checks; do the same for the Level Zero debs and SDK zip.
- [ ] **Hash the ROCm wheels.** `stable.repo.amd.com/rocm/whl-next/` is a moving index; versions
      are constrained with `==`, but hashes are absent. Generate a complete hash-pinned
      requirements file for `rocm[libraries,devel]` using `pip download` / `pip hash`, then use
      `pip install --require-hashes`.
- [ ] BoringSSL's tag pin (`BORINGSSL_VERSION` in upstream's
      `vendor/cpp-httplib/CMakeLists.txt`) remains an upstream follow-up; no downstream override.

### 5. Layer A: model-backed plain-C smoke (B.A)

- [ ] Set `JLLAMA_C_ABI_MODEL` in a suitable CI job and exercise the model-backed half of
      `c_abi`. The existing plain-C compile/smoke runs without a model; Java/C++/FFM inference
      coverage does not exercise this particular consumer. macOS remains a static build.

### 8. Cached/uncached reproducibility measurement (C.3)

- [ ] Compare native hashes from cached and uncached builds of the same commit before considering
      a reproducibility job as a replacement for disabling the shared cache on release tags.
      The release-tag cache bypass is already implemented.

### 11. Windows code signature — deferred by the owner (C.4, C.5)

The signing implementation and `--require-signed` checks exist. A successful
`sign-windows-natives` job without the `sign_windows` switch is an unsigned pass-through and
does not prove Authenticode signing. The owner selected SSL.com IV + eSigner; macOS remains
ad-hoc signed with GPG/Sigstore provenance, without an Apple Developer subscription.

- [ ] **Owner:** obtain the SSL.com IV certificate, enroll it in eSigner and verify the quota
      against **35 DLL signatures per run**. Configure `ES_USERNAME`, `ES_PASSWORD`,
      `CREDENTIAL_ID`, `ES_TOTP_SECRET` in `maven-central`; keep the TOTP secret scoped to the
      signing job. Recheck pricing before ordering (the original estimate was ~129 USD/year
      for the certificate plus ~15–20 USD/month, metered, for eSigner).
- [ ] Dispatch a first signed run on `main` with `sign_windows` and without
      `publish_to_central`. Verify the publisher on an unpacked `jllama.dll` with
      `Get-AuthenticodeSignature` / `signtool verify /pa /v`, signed Windows smoke artifacts and
      `package`'s `--require-signed` check; then verify a signed snapshot publication.
      `ES_ENDPOINT=https://cs-try.ssl.com` can prove the sandbox flow, but its certificate is
      not trusted by Windows. Unsigned delivery remains possible under the existing owner decision.

## What still has to be measured

Kept as a list because each entry names the machine it needs -- the harness is
[`docs/benchmarks/ec2-harness.md`](docs/benchmarks/ec2-harness.md) and a run is cents:

| what | needs | why it cannot wait for CI |
|---|---|---|
| **OpenVINO on an Intel GPU/NPU** | suitable Intel device, unavailable to these sessions | the 21.7x/336x figures are CPU-plugin fallback only; they do not prove GPU/NPU execution |
| the two **SME** aarch64 modules | Linux aarch64 with SME, absent from current CI/cloud machines | macOS's static Metal library does not exercise these modules |
| **gpt-oss-20b on both MoE paths of `llama/patches/0022`**, on the **Core Ultra 7 165U** (`-t 6/8/10`) | the laptop; switch with `LlamaModel.setCpuMoeBatched` or `GGML_CPU_REPACK_MMID_BATCHED=0` ([`docs/benchmarks/moe-mxfp4-cpu-alderlake.md`](docs/benchmarks/moe-mxfp4-cpu-alderlake.md)) | upstream's static work split meets real P-, E- and LP-E-cores only there; the sandbox could emulate one slow core for prompt processing (−49 % upstream's path, −24 % batched) but not for generation |
| **`llama/patches/0022` with real weights on the CPU families still missing** -- done on Zen 3 (AVX2, `haswell`), a Neoverse-N1 (dotprod, `armv8.2_2`) and an IBM z16 (VXE, `z16`), with gpt-oss-20b, Qwen3-30B-A3B Q4_K_M and OLMoE Q4_0 ([`docs/benchmarks/moe-repack-real-weights.md`](docs/benchmarks/moe-repack-real-weights.md): +13 % to +66 % prompt processing, generation unchanged, KLD 0 for MXFP4/Q4_0); the same `llama-bench` + `llama-perplexity --kl-divergence` pair, both switch positions, on the rest | x86-64 with AVX-512/AMX and Zen 4 (EC2 or Vultr, paid: the `sapphirerapids` and `zen4` modules); aarch64 with i8mm/SVE (Graviton 3/4, paid: the `q4_K_8x8` / `q4_0_4x8` kernels) | each module calls other gemm kernels; the N1 run found a defect (Q4_K −18 % at 32 threads, fixed) that x86 did not show. The C++ guard (`test_moe_repack`) still skips on Windows arm64 and s390x, where nothing is repacked in the test build |
| [`docs/local-test-plan.md`](docs/local-test-plan.md) parts A–D on the **Core Ultra 7 165U** | the laptop the Windows guide (#575) is written for | part A -- which CPU module ggml picks there and what it gains -- is what CI cannot measure; the only report so far is the Zen 3 one (`docs/handover/local-agent-report-b11538-windows.md`). One result file per machine, unmeasured values marked as such |

## Open — jllama-specific

### Deferred API features from the b11800–b11827 range

- **Server-side TTS through `LlamaModel` / the Java transport.** `POST /tts` (b11781, #26603) runs
  `SERVER_TASK_TYPE_TTS` on a chat model loaded with a TTS mmproj and streams `server_task_result_tts` chunks
  (`audio`, `sample_rate`, `final`; WAV or PCM per `out_type`). `NativeServer` serves it already (classic and
  attach mode, the route is in upstream's `server_register_routes()`, which `0007` registers). A `LlamaModel.tts(...)` over the C API and a
  `/v1/audio/speech`-shaped route on `OpenAiCompatServer` would make it reachable without the native
  frontend; `TextToSpeech` (its own context over `mtmd_helper::gen_audio`) stays the one-shot path. Not before
  the release.
- **Multimodal rerank through `LlamaModel` / the Java transport.** Since b11820 (#29625) upstream's rerank
  route takes `{"text": ..., "image": ...}` items and splices the images through `process_mtmd_prompt`
  (`format_prompt_rerank(..., query_files, doc_files, ...)`, `handle_media()` for the decoding); `NativeServer`
  serves it. `engine::rerank` passes empty file lists, `jllama_rerank` and `LlamaModel.rerank` take strings,
  and `OpenAiCompatServer` refuses an `image` with a `400` naming `NativeServer` rather than ranking the text
  alone. Wiring it means items in the C API (text plus base64 media, as the multimodal prompt of
  `jllama_completions` -- `multimodal_data` -- already carries) and a typed overload on `LlamaModel`; it
  needs a multimodal reranker in the CI model set to be tested. Not before the release.
- **Realtime transcription** (`/v1/realtime`, WebSocket, b11780, #30636: `mtmd_audio_stream`,
  `SERVER_TASK_TYPE_TRANSCRIBE_RT`, `SERVER_CONTROL_ACTION_AUDIO_*`). Served by `NativeServer`; the JDK's
  `com.sun.net.httpserver` has no WebSocket, so the Java transport cannot follow without a dependency. Not
  planned.
- **Realtime conversations** (`/v1/realtime`, WebSocket, b11946, #31269: `SERVER_TASK_TYPE_REALTIME`,
  `server_realtime_slot`, the `TEXT_APPEND` / `RESPONSE_CREATE` / `RESPONSE_CANCEL` / `CLEAR` control actions,
  `server_task_result_realtime` with text, audio, `response_done`). A model that answers after the audio (an
  audio-in/audio-out model such as LFM3-Audio) holds a spoken conversation with text and audio out. `NativeServer`
  serves it already (classic and attach mode: `server-realtime.cpp` is compiled in since #590 and the route is
  upstream's), and `RealtimeNativeServerIntegrationTest` drives it end to end with LFM3-Audio over the JDK's
  WebSocket client: session formats, a text and an audio response, spoken input, a transcription session and a
  tool call (b11960, #31287), 6 cases in ~12 s on 16 cores (measured locally; in CI since 2026-11-10 with the `realtime` rows). Missing is a Java API: a session type on `LlamaModel` over the C API -- push PCM, append text,
  request a response with an output mode (`MTMD_HELPER_GEN_AUDIO_OUT_AUTO` / `TEXT` / `AUDIO` / `MIXED`), receive
  text and PCM events, cancel, clear -- and on the Java transport the same as for transcription above: the JDK's
  `com.sun.net.httpserver` has no WebSocket. The model to test it with is in CI's set (LFM3-Audio, feature `realtime`). Large; decide when
  someone asks for spoken conversations from Java.
- **`--connect` / `--connect-code`** (b11744, #28277): the server spawns the separate `llama-connect`
  binary (WebRTC tunnel to a browser outside the LAN). `LLAMA_CONNECT` stays OFF (a prebuilt download at
  configure time), `tools/server/server-connect.cpp` is compiled because `server.cpp` references it, and the
  flag answers with upstream's "not available" message. No plan to bundle a binary.

### From the b11968–b12242 bumps (2026-11-09/10) -- to implement, to decide, to measure

What the range brought that this library does not use yet, or that changed under it. Done at the bump
and therefore not listed: predefined TTS voices (`TextToSpeech.getVoices()`, `jllama_tts_voices`, the
`voice` of a synthesis handed over as upstream now wants it), `ModelParameters.addShard`
(`--shard`), `setMoeCacheCpuShare` (`--moe-cache-cpu-share`).

**Windows, non-ASCII paths: measured, the slot files fixed, the download paths open.** Since b12131 /
b12132 (#31682, #31681) upstream builds its `std::filesystem::path` and `std::fstream` objects in
`common/` and `tools/server/` from a plain `std::string`, which Windows reads in the process's
active code page; its own executables are UTF-8 there through an app manifest (#31329), a library
inside `java.exe` is not. Measured on 2026-11-10 on Windows 11, code page 1252, through
`jllama_test.exe` and the C API
([`docs/handover/local-agent-report-windows-non-ascii-paths.md`](docs/handover/local-agent-report-windows-non-ascii-paths.md)):
  - **Fixed:** the slot directory (`fs_is_directory`, "not a directory" at load; `patches/0024`) and
    the checkpoint / draft-state appendix of a slot file (`patches/0025`). Without `0025` an ASCII
    directory with a non-ASCII file name put the appendix into a second, mis-named file: the save
    and the restore both reported success and the draft state was gone. An earlier version of this
    entry said that file could not occur; it could, through the file name. Also fixed, in this
    library's own streams: the calibration text of `jllama_imatrix`, the legacy output of
    `jllama_imatrix_merge` and the trainer's `training_file`. `CApiSlotFiles` and
    `CApiNonAsciiPaths` hold all of it, and the paths that worked before (`-m`, `--mmproj`,
    `--lora`, `--shard`, the model and matrix the quantizer reads, the GGUF outputs of the imatrix
    run and the trainer).
    `patches/0026` adds the last reader: a legacy imatrix file under such a path
    (`common_imatrix_load_legacy`, a narrow `std::ifstream`), `patches/0027` `--log-file`
    (`common/log.cpp`, a narrow `fopen`) and `patches/0028` the quantizer's output
    (`src/llama-quant.cpp`, a narrow `std::ofstream`), and `patches/0029` a `file://` media URL
    below `--media-path` (`handle_media`, found by the review of PR #612). **Measured on 2026-11-11 on the same machine
    ([`docs/handover/local-agent-report-windows-non-ascii-paths-b12283.md`](docs/handover/local-agent-report-windows-non-ascii-paths-b12283.md)):** each of `0025`-`0030` turns exactly its own test red when
    it is removed, and all are green with the 20 patches.
  - **Open, narrow sites in llama.cpp, a non-ASCII path fails or lands under a mangled name:** the
    download paths (`-m` as the target of `--model-url`, `LLAMA_CACHE`; several files under
    `common/`). README names them. A patch, or upstream: not decided.
  - **Fixed, not a file site:** a value llama.cpp takes from an environment variable
    (`LLAMA_ARG_SLOT_SAVE_PATH`, `LLAMA_ARG_LOG_FILE`, ...) was what `getenv` returns, the ANSI code
    page inside `java.exe`, and failed the load before any argument could override it;
    `patches/0030` reads it as UTF-16 and converts. Measured with the others (without it: "No mapping for the Unicode
    character exists in the target multi-byte code page").
  - **Not measured:** `--models-dir` in router mode, `-hf`, the MCP configuration file and the
    built-in file tools. The Java layer was measured on 2026-11-11 with
    Temurin 25 (model, `--log-file`, slots in both shapes, quantizer, imatrix, legacy merge: all work;
    `NonAsciiPathIntegrationTest` holds them on the Windows Java jobs).
  - **Open, found by that measurement: an extraction directory with a character outside the ANSI code
    page cannot be loaded from.** `java.io.tmpdir` / `net.ladenthin.llama.tmpdir` under `tmp-日` on code
    page 1252: Java extracts the files, `System.load` then names `tmp-?` and fails with "Can't find
    dependent libraries" (the JVM passes the path in `sun.jnu.encoding`; read from the message, not
    from the JDK source). Within the code page (`ü` on 1252) it works. Possible in `LlamaLoader`:
    detect a directory the encoding cannot represent and fall back to another one or to its 8.3
    name. Not implemented. The error names the cause since 2026-11-11
    (`LlamaLoader.unencodablePathHint`: the path is checked against `sun.jnu.encoding` when a load
    fails; unit-tested, not run on Windows).
  - **Open, found by that measurement: a model's `--log-file` stays attached after `close()`.**
    `common_log_main()` is process-wide, so later models of the same JVM keep writing into the file
    and Windows cannot delete it while the JVM lives. Not a path defect. The C API test detaches it
    with `common_log_set_file(..., nullptr)`; Java has no call for that.
  - **A second build directory rewrites the shared natives directory's provenance.** A configure of
    another build tree (the counter-checks' `build-no00NN`) writes `jllama-build.txt` and
    `jllama-sbom.json` into `src/main/natives/...` too, so they then name that tree's patch set;
    reconfigure the main build directory afterwards.
  - Options if it reproduces: (a) a patch that restores `u8path()` at the affected sites (about 30
    lines over `common/{common,download,hf-cache,preset}.cpp` and
    `tools/server/server-{context,mcp,models,tools}.cpp`; upstream removed exactly these, so it has
    to be refreshed whenever they move); (b) refuse such a path in the Java layer on Windows with a
    message; (c) report it upstream as the embedding-host half of #31329 -- a library cannot carry a
    manifest. `setlocale(LC_ALL, ".UTF-8")` would switch MSVC's narrow file APIs to UTF-8 but changes
    the C locale of the whole JVM (the UCRT is shared), so it is not an option to take silently.

**Found while writing tests for the bump (2026-11-10), older than it; fixed the same day**
- A draft model alone did not speculate (upstream's types default to `none`): the engine now uses a
  draft model as `draft-simple` when no type is given, and `ModelParameters.setSpecTypes` writes
  `--spec-type`. `LlamaModelTest#testSpeculativeDecoding` asserts `getDraftTokensTotal() > 0`
  (model-backed, so its first run is CI's: should CodeLlama and AMD-Llama-135m not draft together,
  that is the finding).
- Enum constants no receiver accepts are removed: `PoolingType.UNSPECIFIED`,
  `RopeScalingType.UNSPECIFIED` / `LONGROPE` / `MAX_VALUE`, `QuantizationType.NVFP4`.
- The quantizer takes an importance matrix (`jllama_quantize_imatrix`, `LlamaQuantizer`, `Jllama`),
  nine types need one (`QuantizationType.requiresImportanceMatrix()`), and
  `LlamaQuantizer.computeImportanceMatrix` / `jllama_imatrix` compute it, with a progress callback
  that can cancel. Left out of the port of upstream's `tools/imatrix`: the activation sums
  (`.in_sum`, statistics only), `--process-output`, the perplexity report, intermediate saves, and the tool's
  `--include-weights` / `--exclude-weights` on the quantizer side. **Compared with upstream's
  `llama-imatrix`** (b12242, built standalone, the dense and the mixture-of-experts random model,
  the same text and chunk size): the same tensor names, every `.in_sum2` and `.counts` value
  identical (worst relative difference 0.0), the same `imatrix.datasets` / `chunk_count` /
  `chunk_size`. Upstream's file additionally holds an `.in_sum` and a `.stats` tensor per weight and
  `imatrix.stats_schema`, which the quantizer does not read. The comparison was done by hand; it is
  not a test, because no CI job builds upstream's tools. **Merging and the legacy format are in**:
  `jllama_imatrix_merge` / `LlamaQuantizer.mergeImportanceMatrices` add up matrices of one model
  (GGUF or legacy in, GGUF or legacy out; one input converts). Different chunk sizes are refused
  where upstream warns and goes on.
- Rank pooling on a model without a classification head is refused at load (upstream answers the
  first value of the pooled embedding as the score). The check reads the tensor list of the model
  file and gives a split model the benefit of the doubt.
- The random test models cover a decoder (`llama`), a mixture of experts, an encoder with an optional
  classification head (`bert`), a decoder with an MTP block (`hy_v3`), and six vocabularies: bytes,
  BPE with merges, SentencePiece without and with byte pieces, Unigram (`t5`) and RWKV. Not built:
  the PLaMo vocabularies, an encoder-decoder (T5 itself: the Unigram vocabulary sits on the `llama`
  decoder here), RWKV as an architecture, the hybrid architectures.
- **A SentencePiece vocabulary without byte pieces cannot tokenize a character it has no piece
  for**: llama.cpp's lookup of the byte piece throws and the message is the map's
  (`unordered_map::at`). An error and not a crash, pinned by `CApiTokenizers`; a converted
  SentencePiece model always has the byte pieces, so this is a trap of hand-made vocabularies. The
  message could name the cause (catch `std::out_of_range` around the tokenization in the engine).
- **A SentencePiece round trip comes back behind a space** when the tokens carry no `<s>`
  (`detokenize(tokenize("cat"))` is `" cat"`): upstream's behaviour, pinned by `CApiTokenizers`.
  Decided: `LlamaModel.decode` says so in its Javadoc and stays what upstream answers; a caller who
  needs the text back strips the one leading space.
- **A model with MTP layers does not draft unless `--spec-type draft-mtp` is given.** Decided: it
  stays explicit (`ModelParameters.setSpecTypes(SpecType.DRAFT_MTP)`, said in its Javadoc). A draft
  model is a file the caller names, so using it is what was asked for; MTP layers come with the
  model, and turning them on by default would change the speed and memory of every load of such a
  model without a request for it -- upstream's own default is `none` for the same model.
- **A grammar aborts the process when the model's end-of-generation byte is part of the generation
  prompt** (`GGML_ABORT`, `llama-grammar.cpp:1527`), met while writing `CApiStructured`: the random
  model's default EOS is the byte `>`, which the chat template's prompt contains. The fixture uses
  another EOS (4, a control token). A real tokenizer does not have a printable EOS, so this is a
  trap of the test models, not a defect to report.
- **The quantization sweep takes 9 s on x86-64, and the model is not why.** Measured per type:
  IQ1_S 2.8 s, IQ3_XXS 1.6 s, IQ2_XS 1.1 s, every other type below 0.7 s -- the one-time
  construction of ggml's IQ codebook tables, which no smaller model avoids. Under qemu
  (`test-cpp-linux-s390x`) that will be minutes; watch its first run.
- `setMoeCacheMib` on a host without a GPU fails the load ("MoE cache requires a GPU backend"); its
  Javadoc is corrected.

**Build and CI**
- **CUDA 13.5: built, not measured on a GPU.** Both CUDA jobs install 13.5 and no longer pass
  `-DGGML_CUDA_CCCL_VERSION` (13.5 bundles CCCL 3.5), as upstream's release builds since b12169
  (#31788). Run 42783058488 (2026-11-10) built both modules: Linux `cuda-toolkit-13-5-13.5.0-1`,
  Windows `redistrib_13.5.0.json`, which `CUDA_REDIST_MANIFEST` pins now. The Linux job's nvcc
  passes went through sccache without an error or a retry on a cold cache (2622 misses, the C/C++
  TUs 100 % hits, 100 minutes); the next run's table says whether the warm cache pays. Open:
  - whether `option(BUILD_TESTING ... OFF)` before the first `FetchContent_MakeAvailable()` is still
    needed now that ggml no longer fetches CCCL (it stays; removing it is a change of its own);
  - `jllama-sbom.json` names the bundled CCCL from the toolkit's `cuda/std/__cccl/version.h`
    (`CCCL_VERSION`, one number). The probe is tested with CCCL's real header in a stand-in
    directory, not against an installed toolkit; `merge-native-artifacts.sh` fails `package` when
    a CUDA artifact's BOM names no CCCL version, so the first run says whether the path is right.
  The driver floor moves with the toolkit for GPUs newer than Blackwell only (`121-virtual`, the one
  PTX-only target); README says 13.5.
- **The SYCL modules build the XMX (DPAS) code paths now** (`GGML_SYCL_SUPPORT_XMX`, default ON,
  b11980 #31125). Nothing to change; read the first Publish run's two SYCL jobs for a compiler that
  rejects the new kernels, and the module's dependency list for a new import.
- **`ggml-base` is compiled with `-fp-model=precise` under `icx`/`icpx`** (upstream's own change in
  `ggml/src/CMakeLists.txt`). A module-only SYCL build links the ggml-base of its own configure but
  ships only the module, so nothing changes in a jar.
- **mtmd is a unity build** (#31760, batches of 16 model sources). Follows by itself; if a
  Windows/sccache job shows a miss rate for `mtmd` it did not have, that is why.
- **RPC over more than two ranks / NCCL** (b12079 #28967): the CUDA module would link NCCL only if
  the toolkit image has it (`find_package(NCCL)`, unchanged). Nothing ships it; `RpcServer` serves
  the pairwise path. Only relevant if someone wants multi-node tensor parallelism from Java.

**API surface not exposed yet**
- **New audio generation pipelines: Chatterbox (b12097 #30993) and Drama-3 (b12116 #31613)**,
  `MTMD_GEN_AUDIO_TYPE_CHATTERBOX` / `DRAMA3`. `TextToSpeech` drives `mtmd_helper::gen_audio`
  generically, so they are expected to work; neither was loaded. `TtsPipelineIntegrationTest` is
  the test (see "Tests that exist and have not run" below).
- **Nemotron 3.5 Transcribe (b12087), MageVL-Cowork (b12192), Qwen3-VL's video prompt format
  (b12098, `MTMD_VIDEO_TEMPLATE_QWEN3VL`)**: upstream-side model support, reached through the
  existing multimodal paths. No code here; candidates for `AudioInputIntegrationTest` /
  `MultimodalIntegrationTest` models.
- **Reference-counted prompt checkpoints** (b12141 #30137) and **the MoE expert cache's CPU share**
  (b12191): both internal. Whether upstream exports a hit rate
  for the cache that `ServerMetrics` could carry was not looked at.
- **Speculative decoding state** (b12091 #30141): covered. `CApiSlotFiles` saves and restores a slot
  with a draft model, `CApiMtp` one of a model drafting with its MTP block (the restored slot
  answers the same request the same way).

**From b12242 to b12283 (2026-11-11)** -- 41 upstream commits; 15 patches unchanged, `0015`
reduced, `0027` and `0028` new.
- **8-bit float KV cache** (`--cache-type-k/-v f8_e4m3`, #31922): `CacheType.F8_E4M3`; the enum
  sweep loads the random model with it and answers on the CPU. Not measured: memory and quality
  against `q8_0` on a real model, and any GPU backend.
- **`--tts-lang` / `--tts-voice`** (#32022, the server's defaults for `/tts` and
  `/v1/audio/speech`): `ModelParameters.setTtsLanguage` / `setTtsVoice` for `NativeServer`, and
  `OpenAiServerConfig.speechLanguage` / `speechVoice` for `OpenAiCompatServer`
  (`OpenAiCompatServerSpeechTest`, model-free). The standalone launcher has them as
  `--tts-lang` / `--tts-voice` next to `--tts-model` + `--tts-mmproj`, which load a `TextToSpeech`
  beside the model (`OpenAiServerCliTest`); that the launcher then really speaks was not run with
  a speech model here -- no test starts `main` with one.
- **Realtime output voice and language** (`session.update` `audio.output.voice/language`, refused
  once the conversation started): `RealtimeNativeServerIntegrationTest` has the case, **written
  without the model**; its first run is this bump's Publish run.
- **ggml-rpc client half of `patches/0015` is upstream's** (#30415). A device lost during a
  computation still aborts the process; the server half (stop, COMM_INIT bind address) stays ours.
- **SYCL toolchain 2026.2** (Deep Learning Essentials 2026.2.0, oneDNN 2026.1.0, Level Zero
  1.34.0, as upstream's release jobs). The run-time DLL names of the Windows module (README names
  `sycl9.dll`, `mkl_sycl_blas.6.dll`) are read off the package job's import list after the first
  run, not known before it.
- **WebUI**: 22 more npm packages (CodeMirror / Lezer, MIT), license texts regenerated.

**Generated models for paths no CI model reaches (2026-11-11)** -- four additions to
`tiny_llama_gguf.hpp`, each with a C API test that runs in every `C++ Tests` job:
- a sliding-window decoder (`cohere2`), whose slot file carries context checkpoints -- the
  checkpoint half of `patches/0025`, unmeasured until now;
- a recurrent model (`mamba`): cached prompt and slot save / erase / restore;
- a decision model (`openjev`) for `/v1/systemone`;
- an ultravox-style audio projector: a WAV clip in a raw prompt and in a chat message.
`SystemOneIntegrationTest`'s decision half runs in CI since upstream's `tinylaya-for-testing-Q8_0.gguf`
joined `models.csv` (feature `decision`, 97 MB, Apache-2.0, decision type `laya`; **not run locally**:
Hugging Face was not reachable from the sandbox it was added in, so the first Publish run is its
measurement). Open: `AudioInputIntegrationTest` still skips itself everywhere -- an audio pair is
~1 GB, or the generator's files are written by a build step the Java tests can read. Not decided.
And one finding for upstream, not
filed: an ultravox projector whose stack factor does not divide 1500 aborts the process on the first
clip ("Invalid number of output tokens", expected 187, got 188) -- whether a real ultravox GGUF
(stack factor 8) reaches that was not checked.

**From b12233 to b12242 (2026-11-10)** -- nine upstream commits, all 13 patches unchanged, no change
in the request fields, their limits or the response keys.
- **`POST /tts` and `/v1/audio/speech` on `OpenAiCompatServer`**: served (`SpeechBackend`, over a
  `TextToSpeech` of its own; `/tts` with `speaker_ref_b64`, `n_predict`, `top_k`, `seed`,
  `temperature`, `pcm` as float and `pcm16`). Left out: `stream` (refused -- `TextToSpeech` has no
  streaming entry point, the audio is answered in one piece), a multipart `speaker_ref` upload, and
  `top_p` / `repeat_penalty`, which `TextToSpeech` does not take.
- **`mtmd_helper_resampler`** (b12236 #31964), a streaming resampler for mono F32 PCM. Upstream uses
  it for realtime input at another rate than the model's. A use here: `TextToSpeech` takes a
  speaker reference only as a file upstream decodes; a caller holding PCM at another rate has no
  entry point. Only worth it with a PCM overload of `synthesize`.
- **Gemma 4 audio input as a stream** (b12237 #31965): upstream-side, through the realtime route of
  `NativeServer`. A candidate model for `AudioInputIntegrationTest`; nothing to change here.
- **`response_format` next to `tools`** (b12242 #31956): `CApiStructured` pins the whole answer
  being constrained, hermetically, and `ToolCallingIntegrationTest` with the tool model (first run:
  CI's). Whether the hermetic test is red before b12242 was not measured (the build tree was already
  at the new tag).
- **The realtime route's `instructions`, `output_modalities` and `max_output_tokens`** in
  `session.update` (b12239 #31978): `RealtimeNativeServerIntegrationTest` has the cases (the
  session reports them back, a response without its own settings is text only; an unknown modality
  and a bad limit are error events). **Written without the model**: the class is property-gated
  (LFM3-Audio, in CI's set since 2026-11-10); they ran green in Publish run 42853242881.
- **Speech and realtime tests whose models joined the CI set on 2026-11-10** (written in a sandbox
  that cannot reach huggingface.co; their first run was Publish run 42853242881, green on all six
  Java test jobs):
  `PocketTtsIntegrationTest` and `ChatterboxIntegrationTest` (the pipeline contract of
  `AbstractTtsPipelineContract`: sample rate from the header, every listed voice, a speaker
  reference cloned or refused as the pipeline reports, the temperature accepted),
  `TtsChatTemplateIntegrationTest` and `RealtimeNativeServerIntegrationTest` (LFM3-Audio). The
  rows are `pockettts`, `chatterbox` and `realtime` in `.github/models.csv`, 2.7 GB more for every
  job that fetches the whole set; the GGUF headers were checked against b12242's loaders, no model
  was run. Pocket TTS comes from a community repository (`macolomb/pocket-tts-without-voice-cloning`,
  cc-by-4.0: neither ggml-org nor Kyutai publishes a GGUF pair with predefined voices).
  **Drama-3 stays out**: the smallest pair is 14.6 GB (`mradermacher/drama-3-GGUF`, Q2_K backbone
  12.9 GB + Q8_0 mmproj 1.7 GB, research licence), more than a hosted runner should download in
  six jobs; `TtsPipelineIntegrationTest` runs the same contract on it by property
  (`-Dnet.ladenthin.llama.tts.pipeline.model/.mmproj`). The rows, should a runner for it exist:
  `drama-3.Q2_K.gguf` sha256 `10c87623811b8948806801c8263b3d896f3cd751067d14eeddf5ba6c6732fc75`,
  `drama-3.mmproj-Q8_0.gguf` sha256 `dc7fe995f67a68f375d86b8f0ad6fe2d8b4761210a73dca9bc49768761edcc2f`,
  both at commit `690a670e47fb6bdd3ff5683b9f7dca9338aad448`.

**Compatibility notes to carry into the release notes**
- **RPC protocol 9** (b12217): `RpcServer` and a `--rpc` client must run the same version of this
  library.
- **`TextToSpeech` voice handling** changed for models with predefined voices (first voice by
  default, voice + speaker reference refused).
- **`HF_TOKEN` is sent to `huggingface.co` hosts only** (b12166 #31730): a mirror on another host
  that needs the token no longer receives it.

### Modular natives -- follow-ups

1. **Windows GPU modules on real hardware -- the tg128 of the CUDA module.** Device registration,
   the `device_id` dedup (CUDA wins), the iGPU, OpenCL and the README sentence on device names were
   measured on 2026-10-26 on the published 6.0.0 jars
   ([`docs/benchmarks/gpu-module-jars.md`](docs/benchmarks/gpu-module-jars.md), section 3). Still open:
   the tg128 of the CUDA module against the static build's 265.6 t/s (`GGML_CUDA_GRAPHS_DEFAULT` is
   repeated in the modular path, so the 88.8 of addendum 3 of
   `docs/handover/local-agent-report-b11538-windows.md` should be gone). That figure used another
   model, so it needs the same model, `llama-bench`-style, in both builds.
2. **CUDA for Windows arm64 (`cuda13-windows-aarch64`) -- upstream builds it, we do not.** Their
   `windows-cuda` job is a matrix over `x64` and `arm64`, the latter through
   `cmake/arm64-windows-msvc-cuda` plus the arm64 `cudart`/`cublas` redistributables. This project has
   `cuda13-windows-x86-64` only. The arm64 Windows toolchain already exists here (the `cpu` and
   `vulkan` jars for that platform), so the missing piece is the CUDA toolkit's arm64 components and
   the toolchain file. Niche hardware today (Windows-on-ARM with a discrete NVIDIA GPU), but it is a
   straight coverage gap against upstream rather than a judgement call, and the "support more rather
   than fewer" policy that governs the ROCm targets applies. **Not** a gap: Linux aarch64 CUDA
   (Jetson, Grace-Hopper) -- upstream does not build that either, so it would be new ground.

3. **A Vulkan module for Android** (`vulkan-android-aarch64`, upstream ships none either): `JNI_OnLoad`
   already tries `libggml-vulkan.so` by soname, so it is one more module-only dockcross job plus the
   natives-jar wiring and an AAR flavor -- if anyone asks.

### sccache — optional upstream and nvcc follow-ups

- **After the release:** propose support for `-Xclang -flto-visibility-public-std` in sccache
  (`src/compiler/clang.rs`, `PassThroughFlag`, with a test of CMake's full sequence). The Windows
  workaround in `llama/cmake/msvc-runtime-gnu-clang.cmake` already permits caching; see CLAUDE.md
  "sccache on every Windows Ninja job". This is independent of the nvcc issue below.
- **The Windows CUDA job is the pipeline's critical path, uncached: 96 min** in run 40217818548,
  nearly all of it nvcc, because `build.bat` does not wrap nvcc (12 compile requests through
  sccache, all C/C++). The Linux CUDA job wraps it again since 2026-10-25 and hit 2640 of 2640
  (CUDA, PTX and CUBIN at 100 %, build step ~12 min) in the same run. Whether sccache 0.19.0 parses
  the Windows nvcc command line (#2470, fixed by #2811 since 0.18.0) is untested: try
  `-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache` in `build.bat` on a branch, with the existing uncached
  retry as the fallback, and read two runs (the first fills the cache).
- **Upstream sccache, open on their side:** mozilla/sccache#2809 (the `-virtual` arch fix, not ours)
  and #2788 (`--offload-compress` in preprocessor cache mode; a nudge after the release).

### RPC backend — follow-ups

- **A server lost mid-inference still aborts the JVM.** Patch `0015` makes *registration* fail
  softly; every call after it (`get_dispatcher()`, `RPC_STATUS_ASSERT` in the dispatcher's `work()`)
  still ends in `GGML_ABORT`, because the ggml backend interface has no error return for a lost
  device (`graph_compute` returns a status, but buffer `set/get_tensor` are `void`). A real fix is an
  upstream change: carry a failed-state flag through the dispatcher, fail the pending futures, and
  surface it as a `GGML_STATUS_FAILED` at the next `graph_compute`, which llama.cpp already turns into
  a decode error. File upstream first; do not carry it downstream.
- **A served device that cannot run an operation aborts the server process.** ggml-rpc's client
  answers every `supports_op` with `true` (upstream `//TODO: call the remote backend and cache the
  results` in `ggml_backend_rpc_device_supports_op`), so the scheduler never falls back and the
  server hits `GGML_ABORT("unsupported op")` in the device's graph compute. Seen on the macOS CI
  runners, whose paravirtual Metal GPU has no `MUL_MAT`: an in-JVM `RpcServer` serving it takes the
  JVM down on the first inference. Mitigated, not fixed: `RpcServer`'s device list (`--device CPU`)
  and the tests serve the CPU. The fix is upstream — forward `supports_op` over the protocol (a new
  command, with a per-op cache on the client) — and belongs there, not in `0015`.
- **Android RPC is untested on a device.** Bionic sockets build (upstream ships RPC in its Android
  release too), but the app needs `android.permission.INTERNET` even for loopback, which the AAR
  deliberately does not declare and the emulator fixture does not have. A loopback test on the
  emulator needs that permission in the fixture's manifest only.
- **RPC smoke on Windows.** `smoke-rpc.sh` runs in both Linux rows of the `smoke-natives` matrix
  (`linux-x86-64`, `linux-aarch64`; the step's `if: runner.os == 'Linux'`); the Windows rows need a
  PowerShell port of the script. The Java `RpcServerTest`/`RpcIntegrationTest` already run on every `test-java-*` job
  (Windows and macOS included), so this is about the published jars, not the code path.
- **Authentication / TLS.** Upstream has none; the documented answer is a trusted network or a
  tunnel. Only worth doing if it lands upstream.
- **RDMA transport** (`GGML_RPC_RDMA`) as its own classifier, since it needs `libibverbs` at runtime.
- **Several clients at once.** Upstream's server serves one connection at a time.
- **Edges of `0015`'s stoppable server, from the review of 2026-10-25 (none seen in CI).**
  `wait_socket()` waits with `WSAPoll`, which before Windows 10 2004 does not report a refused
  non-blocking connect, so a peer connect there waits for the stop instead of failing; checking
  `SO_ERROR` while the wait times out would close that. An `accept()` error other than "would block"
  ends the serve loop, as it does upstream (a retry on `ECONNABORTED`/`EMFILE` would keep it alive).
  `test_rpc_comm.cpp` takes ports from `free_port()` (bind to 0, close, reuse), which another process
  can take in between; a bind failure there is a flaky test, not a defect. Each needs a Windows
  build to verify and belongs into the upstream submission of `0015`.

### Atmosphere coding agent (`llama-atmosphere-agent/`) — follow-ups

- **Tool rounds across REPL turns ride as a note, not as `tool_calls` messages -- and that cannot
  change on this Atmosphere.** The earlier entry here asked to keep the full `ChatMessage` list
  (incl. `tool_calls`/`tool`) per turn; the recorded measurement used `atmosphere-ai` 4.0.72:
  `AbstractAgentRuntime.assembleMessages` rebuilds every history entry as
  `new ChatMessage(h.role(), h.content())` (checked in the bytecode), so the tool-call array and the
  tool-call id never leave the framework. What ships instead: `AgentSession.toolNote` carries each
  turn's calls and results (cut at 400 chars) in front of the next user message, pinned by
  `LocalAgentTest.aToolCallStaysInTheHistorySoTheNextTurnSeesItHappened`; the full placement
  argument is in `llama-atmosphere-agent/CLAUDE.md`, point 8. Reopen only with an Atmosphere
  release whose `assembleMessages` preserves tool calls (then a protocol-faithful replay is the
  better form, and the note goes).
- **In-stream engine errors are swallowed by Atmosphere** (pinned in
  `AtmosphereWireContractTest.midStreamEngineFailureCompletesSilentlyRatherThanErroring`): an SSE
  `data: {"error":…}` after HTTP 200 is ignored by `OpenAiCompatibleClient.processSSELine` (it reads
  only `choices[0]`). Worth an upstream PR to Atmosphere; until then the console shows an empty turn.
- **Spring Boot `@Agent` variant** (WebSocket/SSE UI via `atmosphere-ai-spring-boot-starter` and
  `LLM_BASE_URL`) is expected to work on the same runtime but is not CI-covered; a smoke that boots
  the starter against the scripted `OpenAiCompatServer` would close that.
- **Anthropic Messages surface.** The server also speaks `/v1/messages`; the Anthropic adapter
  (`org.atmosphere:atmosphere-anthropic`) was not tested against it.
- **Model recommendation table** for the agent (which local GGUFs actually complete an
  edit→build→test loop) — needs a GPU host, not CI.

### OpenAI-compatible HTTP endpoint — open follow-ups (Java transport; deprioritized)

The `OpenAiCompatServer` surface itself is shipped (routes, protocol translations, integration
round-trips — see CLAUDE.md "Two server modes"). **Owner priority: the native-transport
`NativeServer` comes first; Java-transport-only items below are deliberately deprioritized.**

- **Streaming raw-completion remainder:** token-streaming Ollama `/api/generate` (translate
  `text_completion` chunks to NDJSON, mirroring the chat→Ollama translator) and Continue's native
  `POST /completion` route in the llama.cpp-native
  streaming shape (`{"content":…,"stop":…}` per chunk). Java-only server wiring.
- **Future *output* modalities (audio / image) — design note, not yet actionable.** The Java
  transport's integration points are isolated (a new `OpenAiBackend.stream*` primitive +
  `OpenAiSseFormatter.*Chunk` per modality). Two future hooks:
  the existing `TextToSpeech` (Qwen3-TTS since llama.cpp b10270 — OuteTTS no longer exists upstream)
  behind an `/v1/audio/speech`-style route; proxying image/audio generation to an external model.
  Keep chunk formatters modality-neutral.
- **Incremental tool-call streaming on the alternative surfaces.** Ollama/Anthropic/Responses emit each
  tool call whole at end-of-stream (`ToolCallDeltaAccumulator`); revisit only if a client needs
  incremental `input_json_delta` / `function_call_arguments.delta` fidelity.
- **Per-model FIM template registry** — only needed if `/v1/completions`-with-`suffix` FIM is exposed;
  `/infill` applies the model's FIM tokens server-side, so low value.
- **Multi-model registry (Java transport).** The native surface has this via router mode +
  `RouterClient`; the Java `OpenAiCompatServer` still advertises/serves a single model id.
- **Manual real-client validation.** Server-side round-trips exist for every surface; what remains is
  pointing the actual editor clients (Copilot Ollama provider / Custom Endpoint, Claude Code, a
  Responses client) at a running server, since round-trips confirm wire shapes but not each client's
  parser.

### License compliance gate (owner's decision)

The FSFE REUSE check is green on every run. FOSSA analyses the dependencies against the policy the
four Maven repositories share (`.fossa.yml`; rules and the version-specific decisions in
[the dependency license review](https://github.com/bernardladenthin/actions/blob/main/docs/dependency-license-review.md)), and its "License
Compliance" check reports open findings as review requests, not approvals. Which of them block a
release, and which get a documented component decision in FOSSA, is the owner's call.

### Upstream submissions -- not before the release (policy)

**Nothing is filed upstream before the release is out**, neither issues nor PRs; the drafts are not
written before that either. Everything below is the inventory for that day, so it is not rediscovered.

#### The four issues to file, prepared

These are the ones with **no patch to offer** -- a measurement and a question. The work left on each is
writing it; the evidence exists and is linked. File them as **four separate issues**, since they touch
three different subsystems and a combined report gets triaged as one.

| # | title to use | the evidence to paste | the question for the maintainers |
|---|---|---|---|
| 1 | `ggml_backend_load_best picks the richest CPU module, not the fastest` | the three tables in [`docs/benchmarks/cpu-backend-modules.md`](docs/benchmarks/cpu-backend-modules.md): on Granite Rapids the pick gives up 10-11% of token generation, on Graviton 4 **27% of prompt processing** as well | scoring is by feature level by construction. Is a throughput-aware choice wanted at all, or should a consumer be able to override the pick? There is no downstream lever today -- `net.ladenthin.llama.backend` cannot select a *CPU* module |
| 2 | `ggml-rpc: a server lost mid-inference aborts the process` | `patches/0015`'s header plus the RPC section of `CLAUDE.md`: registration was made non-aborting downstream, every path *after* registration still ends in `GGML_ABORT` | the ggml backend interface has no error return for a device that went away. Is adding one in scope, or is aborting the intended contract for a lost device? This is the blocker for using RPC from a long-lived host process |
| 3 | `ggml-rpc: the client answers every supports_op with true` | the RPC section of `CLAUDE.md`: a served device that lacks an op **aborts the server**. Reproduced on the macOS CI runners' paravirtual Metal GPU, which has no `MUL_MAT` and took the first run down; it is why every test here serves `CPU` by name | upstream's own TODO comment admits it. Should the client forward the query to the server, or should `rpc-server -d` refuse a device whose op set it cannot report? |
| 4 | `windows-rocm: 83,362 compiler warnings, against 9 on linux-rocm` | the table below in this section (714,345 log lines against 2,455; the per-class counts; all of them in `ggml/src/ggml-cuda/*.cu`) | the cause is the language, not the arch count: HIP sources are compiled as plain C++ on Windows (`CXX_IS_HIPCC` on `WIN32`) and as HIP on Linux. Fix the warnings, or the C++-on-Windows choice? Their own `windows-rocm` job pays the identical cost |

#### Submission rules and remaining patches

**Two rules for the day.** State the measurement and ask the question; do not propose the fix for 1-3,
because each is a design decision in code this project does not own. And say plainly what was done
downstream instead (the opt-in, the non-aborting registration, the `-w` on `ggml-hip`), so a
maintainer can see what a merged change would let this project delete.

There are **thirteen** patches today (`0002`, `0003`, `0006`–`0008`, `0014`, `0015`,
`0017`, `0018`, `0020`–`0023`). Each
accepted upstream change deletes a patch from the bump checklist once the pin is bumped past it.

- **Submittable as they are** (self-contained, no jllama specifics): `0002` (preserve the caller's load-progress callback), `0006` (embeddable
  `llama_server`), `0007` (`llama_server_attach`), `0008` (`LLAMA_SERVER_WORKER_CMD`), `0014` (`common_log_set_callback`), `0015`
  (non-aborting RPC registration, stoppable server, transport fixes), `0017` (x86 prefetch helper),
  **`0018`** (the AMX buffer released with `free()` instead of `ggml_aligned_free` -- one line, with
  the 6/6-against-0/6 measurement in its header and a configure-time guard that is red without it;
  the clearest submission of the set, since the defect reproduces on any AMX machine and the fix
  copies the line `ggml-backend.cpp` already has), `0020` (validate leaf allocations before
  reusing a graph allocation), `0021` (invalidate the slot prompt cache when server LoRA adapters
  change; both hunks have independent drop-checks in `CLAUDE.md`), `0022` (batch an expert's tokens
  through gemm in the repack `MUL_MAT_ID` and hand its work out dynamically -- submitted **without**
  its switch, which exists for measurements here; the evidence is
  `docs/benchmarks/moe-mxfp4-cpu-alderlake.md`), `0023` (IBM Z: drop the duplicate `<vecintrin.h>`
  include that breaks clang, and a scalar `z13` variant below upstream's `z15` -- two questions,
  since upstream documents GCC for s390x and may prefer its own baseline name).
- **Permanent:** `0003` carries upstream PR #22393, which upstream closed without merging.
- **Two upstream-only defects worth an issue on that day, no patch here:** a lost RPC server
  mid-inference still aborts the process (the ggml backend interface has no error return for a lost
  device), and ggml-rpc's client answers every `supports_op` with `true`, so a served device without
  an op aborts the server (both under "RPC backend" above). The separate sccache flag proposal is
  listed under "sccache — optional upstream and nvcc follow-ups"; the local workaround already
  permits plain-clang Windows caching.
- **A fourth, measured 2026-10-11 and again 2026-10-12: the Windows HIP build produces 83,362
  compiler warnings, and the Linux one produces 9.** The full measurement came from run
  38172107721, where the Windows ROCm job's log was
  **714,345 lines with 83,362 warning messages** against **2,455 lines and 9 warnings** in the Linux
  ROCm job -- the same upstream sources, the same TheRock clang, the same 23 GPU targets. It is
  **60x larger than the next biggest job of that run** and carries **98.5%** of all the warnings its
  Windows jobs emit:

  | job | log lines | warnings |
  |---|---:|---:|
  | **Windows ROCm/HIP** | **714,345** | **83,362** |
  | Windows SYCL | 11,981 | 724 |
  | Windows CPU (clang) | 2,627 | 407 |
  | Linux CUDA | 3,266 | 9 |
  | Linux ROCm/HIP | 2,455 | 9 |
  | C++ Tests Windows | 1,655 | 66 |
  | Windows arm64 | 1,499 | 24 |
  | Windows CUDA | 1,068 | 0 |

  The cause is **the language, not the arch count**: upstream compiles the HIP sources as plain C++
  on Windows (`CXX_IS_HIPCC` on `WIN32`) and as the HIP language on Linux, and clang's C++ frontend
  reports what the HIP one does not -- `-Wsign-compare` 29,673 times, `-Wnested-anon-types` 17,964,
  `-Wmissing-noreturn` 16,378, then unused variables, missing braces, unused parameters, cast-qual
  and unreachable breaks. **Every one is in `ggml/src/ggml-cuda/*.cu`**, upstream's own code: not a
  line of ours. Mitigated downstream with `-w` on the `ggml-hip` target, Windows only, which
  subsumes the dllimport suppression; the Linux job keeps every warning, so the information stays
  where it is legible. **Worth an upstream issue with this table** -- the fix is theirs (either the
  warnings or the C++-on-Windows choice), and their own `windows-rocm` job pays the same cost.

- **A third, measured 2026-10-11: `ggml_backend_load_best` picks a slower CPU module than one it
  rejects.** On a Graviton 4 it selects `armv8.6_2` (198.0 t/s pp512) while `armv8.2_2` reaches
  **271.4** on the same machine and model -- the SVE/SVE2/i8mm paths losing to plain DOTPROD+FP16 --
  with generation inside the noise. The scoring function ranks by feature *level*, not by measured
  throughput, so it is by construction rather than a bug, but it costs 37 % of prompt processing on
  the hardware AWS rents most cheaply, and it is invisible to anyone who only runs the module their
  own CPU picks. Worth an issue with the table from `docs/local-test-plan.md`; **no patch here** --
  overriding upstream's scoring downstream would be a maintenance burden with no end.

- **A fourth, measured 2026-10-19 (F187): `--lora-scaled` / `--control-vector-scaled` cannot name an
  absolute Windows path.** `common/arg.cpp` splits each `FNAME:SCALE` item with `string_split(item, ':')`
  and demands exactly two parts, so `X:\models\a.gguf:1.0` is rejected as three. Measured against the
  real parser in `BUGHUNT-MEASUREMENTS.md` M6 (of four argv forms only a relative path and plain
  `--lora` were accepted). The obvious fix is to split on the **last** colon; small enough for a PR,
  so draft it then. Downstream `ModelParameters` refuses such a path with the reason meanwhile.

- **From the branch review of 2026-10-20 (F221): upstream's Anthropic answer names a matched stop
  word in `stop_sequence` but says `stop_reason` `end_turn`**, where the Messages API specifies
  `stop_reason` `stop_sequence`. The fix is one line in each of the two `to_json_anthropic*` functions
  of `tools/server/server-task.cpp` plus an assertion in `test_compat_anthropic.py`; downstream
  `AnthropicApiSupport` corrects it meanwhile. Small enough for a PR, so draft it then.

Dropped-patch decisions, including `0012`, `0016` and `0019`, live in
[docs/history/dropped-llama-patches.md](docs/history/dropped-llama-patches.md).

### llama.cpp upstream feature exposure (queued, deferred by policy)

Add typed API surface only after a real user request. Raw native-server argv already reaches
upstream options; absence of a Java builder method alone does not make a release blocker.

- **JSONL logging:** decide whether the process-wide upstream logger or Java's `args.LogFormat`
  owns the format before exposing `--log-jsonl` / `--no-log-jsonl`; the callback sink is shared.
- **Request-key coverage:** re-derive remaining gaps by comparing `RequestField.values()` and
  the current upstream schema walked by `test_wire_contracts.cpp`. The old b10883 counts are not
  a current API inventory; expose additional fields on demand.
- **Video input (`ContentPart.videoFile(...)` / bytes overload):** the three video configuration
  flags are exposed, but the typed content part and its integration test remain. Emit upstream's
  `input_video` wire shape with base64/data URI/URL as supported by the current parser. Do not
  advertise the untested `imageBytes(..., "video/mp4")` workaround. A real test needs video-capable
  model/projector fixtures and `ffmpeg`/`ffprobe` on the runner, because upstream invokes them.
- **`--spec-synth-len` / `--spec-synth-rates`: deliberately not exposed.** They synthesize fake
  acceptance probabilities for upstream benchmarks. Keep them out of application builder API;
  they remain reachable through raw `NativeServer` argv.
- **Draft backend sampling:** consider
  `ModelParameters.setSpecDraftBackendSampling(boolean)` for speculative-decoding users who
  need to disable `--spec-draft-backend-sampling` explicitly.
- **Runtime reasoning control:** `InferenceParameters.setReasoningControl(boolean)` and a
  corresponding `LlamaModel.endReasoning(...)` action must land together. The flag only arms the
  sampler; the action needs access to the running slot/task. The existing reasoning budget/tag
  fields do not implement this runtime control (F252).
- **Output-slot allocation:** consider a `ModelParameters.setMaxOutputs(int)` mapping for
  `llama_context_params::n_outputs_max` when a consumer needs the memory limit.
- **Multi-Token Prediction:** consider `ModelParameters.setMtp(boolean)` for MTP-trained models,
  following upstream's current speculative-decoding configuration.
- **Suppressed-token inspection:** consider `LlamaModel.getSuppressTokens()` as a read-only
  view of the model's `tokenizer.ggml.suppress_tokens`. Upstream already applies the bias;
  this would expose diagnostics, not fix inference behaviour.

### ROCm Windows — real Radeon runtime verification

The ROCm Windows build uses HIP's clang as a plain C++ compiler, so it takes the hybrid CRT path
(`-fms-runtime-lib=static` plus `/nodefaultlib:libucrt.lib`). Linking succeeds; runtime behaviour
still needs measurement across the `/MT`/`/MD` library boundary, because **ROCm's device libraries
may use the dynamic CRT**. The same class of boundary is why CMake treats SYCL/OpenVINO separately.

The job has been **green** since, so the link holds. What is unmeasured is the **runtime** behaviour
of the resulting module on an actual Radeon GPU -- and that cannot be rented (see the rented-hardware
plan), because the only AMD-GPU instance class has been withdrawn and never officially supported
ROCm anyway. So this needs a real Radeon machine, or a deliberate decision to ship it on the strength
of a green link plus `verify-native-deps.py`'s dependency allowlist. Do not quietly treat "the job is
green" as the measurement.

### Linux s390x -- open

The glibc floor 2.28 (clang-18 against the pypa `manylinux_2_28_s390x` sysroot) and the IBM Z CPU
modules `z13` / `z15` / `z16` are done, measured on real IBM Z (cfarm191, z16) and documented in
CLAUDE.md, "Linux s390x".

- [ ] **z17 (`-march=arch15`)** is skipped: clang 18 rejects the flag (clang 19 on cfarm191 too), and
      ggml's configure drops the variant silently. Re-check when the toolchain's clang moves, and
      measure on a z17 if the farm gets one. Upstream builds `-march=arch15` since b11968 (its zDNN
      release job) with GCC 15.1 / binutils 2.44 on native Ubuntu 26.04 s390x runners (#31317);
      which clang accepts the flag is still to check.
- **Ruled out:** `ubuntu-22.04` as the runner (glibc 2.35 still misses RHEL 8/9, and its cross GCC 11
  cannot build llama.cpp, which needs GCC >= 12). `-static-libstdc++` alone is ruled out too: it
  fixes GLIBCXX but not GLIBC.

### Hardware and test-fixture coverage

- **Real arm64 Android / Adreno OpenCL device:** run the AAR + Kotlin facade and LLM Service app
  on-device. The x86_64 emulator and app UI jobs already run in CI.
- **Kolibri-1 on GPU:** CPU inference and both published GGUF dialects were measured on the real
  78B model; GPU inference remains unmeasured. `0016` was dropped when upstream added Kolibri-1;
  see the retained history and numerical guard, not the former patch, for the current contract.
- **Structural test-constant wiring:** `TestConstantsTest.theShippedModelConstantsGoThroughTheResolver`
  now skips explicitly when the fixture layout cannot distinguish a missing resolver wrapper.
  The `src/test/resources/...` constants are indistinguishable by value even in CI. Consider a
  structural check of the wrapper calls; the resolver itself already has a hermetic fixture test.
- **Windows module measurements:** see "Modular natives -- follow-ups"; the remaining Intel
  GPU/NPU and Linux SME measurements are in "What still has to be measured".
- **Interactive agent:** real console/terminal behaviour needs a person at a terminal; model-backed
  headless integration coverage does not replace that check.

## Open — cross-cutting (slice for this repo)

- **jqwik pin policy** — see [`../workspace/policies/jqwik-prompt-injection.md`](../workspace/policies/jqwik-prompt-injection.md). `jqwik.version ≤ 1.9.3` is mandatory.

- **`@VisibleForTesting` audit.** No usages currently. Walk the production tree for package-private/protected methods or fields that exist purely so tests can reach them, and either annotate (`com.google.common.annotations.VisibleForTesting`) or move into the test source tree.

- **Null-safety refinement.** Review remaining public API surfaces for places where `@Nullable` is more precise than the `@NullMarked` default, respecting the existing `Optional<T>` policy and strict JSpecify/NullAway checks.

- **Review the project-wide `OPM_OVERLY_PERMISSIVE_METHOD` suppression in
  `llama/spotbugs-exclude.xml`.** The package layers now exist. Recheck the findings against the
  current public API before removing the suppression; align with BitcoinAddressFinder and the
  workspace's "OPM scope-tightening" policy. The former "single-root package" premise is stale.

- **Additional ArchUnit rules to consider:** further per-module import restrictions and public-API constraints if useful. Existing layer, Jackson-import and non-final-public-field rules already cover those cases.

- **Cross-repo code-quality TODOs** — see [`../workspace/policies/code-quality-todos.md`](../workspace/policies/code-quality-todos.md) for the canonical `@VisibleForTesting` design-fit review, package hierarchy review, and class/method naming review. This repo has no `@VisibleForTesting` usages today; package and naming reviews remain open.

## GPU backends on rented hardware

Use [docs/rented-hardware-test-plan.md](docs/rented-hardware-test-plan.md) and
[the EC2 harness](docs/benchmarks/ec2-harness.md) for new probes. Recheck current region capacity,
GPU vCPU quotas and prices before launching; the old quota-0 failure predates successful T4
measurements and is not evidence that GPU testing is currently blocked.

| remaining target | hardware needed |
|---|---|
| Windows CUDA/Vulkan/OpenCL coexistence and throughput | real Windows GPU host (RTX 3070 + AMD iGPU target in the local handover) |
| ROCm runtime, including Windows CRT behaviour | actual supported Radeon/Instinct GPU; the retired AWS g4ad is not evidence for current ROCm support |
| OpenVINO GPU/NPU and SYCL GPU execution | suitable Intel GPU/NPU host; CPU fallback does not prove GPU execution |
| `armv9.2_1` / `armv9.2_2` CPU modules | Linux aarch64 with SME; Graviton 4 lacks it, and macOS's static Metal library does not exercise these modules |

## Other open verification and maintenance

- **WER minidumps are switched on but unproven.** #506 makes the Windows test step set
  `Disabled=0` / `DontShowUI=1` under both WER keys next to the `LocalDumps\java.exe` entry
  (`DumpType` 2), because the AMX heap corruption left no dump behind. No Windows crash has happened
  since, so nothing shows that a `.dmp` now lands in the failure artifact. Prove it once rather
  than wait: a throwaway step that crashes a `java.exe` on purpose (an access violation through
  `sun.misc.Unsafe`, or `jshell` writing to address 0) on a branch run, then look for the dump in
  the artifact -- and if none appears, replace the WER route with `procdump -ma -e` around the
  surefire run.
- **Maven plugin policy:** the default release lifecycle plugins (`clean`, `resources`,
  `install`, `deploy`, `compiler`, `jar`) are pinned in this reactor. Still open across the four
  repositories: an explicit `site` version where that lifecycle is used and an enforcer
  `requirePluginVersions` rule, initially warnings-only while aligning the repositories.
- **Unread SonarCloud finding:** inspect the "1 new issue" reported on #486. The quality gate
  subsequently passed, but that does not identify or explain the finding; the sandbox proxy
  previously blocked access to sonarcloud.io.
- **SonarCloud quality gate red: Reliability E on new code** (not a release gate). Checked and false
  positives: S2095 `BatchDispatch.java:126` and S2142 `BatchDispatch.java:146`. Not yet looked at:
  S3077 `LlamaLoader.java:262` and the two S8786 findings. Fix them or mark them in SonarCloud with
  the reason.
- **Three buildcheck unit tests fail on Windows** (green on the Linux runners, also on `main`):
  `test_smokelib.NativesDirectoryTest.test_absolute_set_directory` and `test_relative_set_directory`
  compare a path with `/` against one with `\` (the expectation is built with the OS separator), and
  `test_patches.RenderTest.test_doc_is_current_compares_the_committed_page_with_the_render` fails
  with a bare `False is not true` (cause not found; `docs/patches.md` is LF in the work tree). Fix
  them, or document that the suite is Linux-only.
- **REUSE attribution review:** check the remaining Java/documentation attribution to the original
  fork author with `git blame`, file by file. Attribution is not removed without an explicit owner
  decision. The generated dockcross wrappers already carry dockcross's own copyright attribution.
