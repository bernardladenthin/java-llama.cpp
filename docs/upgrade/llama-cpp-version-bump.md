<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT
-->

# llama.cpp version-bump runbook

This is the **documentation root** for bumping the pinned llama.cpp version. It links the
mechanical edit steps in [`../../CLAUDE.md`](../../CLAUDE.md#upgradingdowngrading-llamacpp-version)
together with a repeatable **target-selection + chunking** strategy so a bump never lands an
unreviewably large diff in one step.

The current pin lives in `llama/CMakeLists.txt` as `GIT_TAG b<nnnn>`. llama.cpp tags **every**
master commit as `b<nnnn>`, but only a subset get GitHub *Releases*.

---

## TL;DR

```bash
# From the repo root. Prints the next reviewable step (b<cur> -> b<next>) and its compare/.patch URLs.
.github/scripts/llama-next-version.sh                 # target = latest RELEASE (atom feed)
.github/scripts/llama-next-version.sh b9900           # target = an explicit tag
```

Then apply the printed `b<cur> -> b<next>` step per [§ Applying a bump](#applying-a-bump) and re-run
the script to walk the next chunk, until it prints **"reaches the latest release — final chunk"**.

---

## 1. Pick the target (topmost release)

The **target candidate is the topmost release** on
<https://github.com/ggml-org/llama.cpp/releases>. Read it from the release **atom feed**, which is
reachable from restricted sandboxes where the ggml-org REST API is blocked:

```
https://github.com/ggml-org/llama.cpp/releases.atom
```

The first `<entry>`'s `releases/tag/b<nnnn>` is the latest release. `llama-next-version.sh` does this
for you; if the feed is rate-limited (repeated unauthenticated fetches can return empty), open the
releases page in a browser and pass the tag explicitly: `llama-next-version.sh b<nnnn>`.

> **Why releases, not just the newest `b<nnnn>` tag:** releases are the versions upstream deems
> shippable; an arbitrary master commit tag may be mid-refactor. Intermediate **chunk** steps
> (below) are allowed to land on non-release tags — they are transient waypoints, not the target.

## 2. Chunk by diff **byte-size**, not commit count

The step size is governed by the **size of `git diff` between the pinned tag and the target**, not by
how many commits separate them:

- If `git diff b<cur> b<target>` is **< 100 KiB**, bump straight to the target in one step.
- If it is **≥ 100 KiB**, pick an **intermediate** `b<nnnn>` tag whose diff from the current pin is the
  largest still **under** the threshold, bump to that first, then repeat. Each step stays a small,
  reviewable patch.

The threshold is a knob (`LLAMA_BUMP_MAX_DIFF_KB`, default `100`). This is a heuristic: diff size grows
monotonically enough with the tag number that the helper binary-searches the intermediate tags safely.

> **`tools/ui` (the WebUI) dominates the full diff** and is *auto-followed* — CI rebuilds the matching
> Svelte UI from the pinned `GIT_TAG`, so it needs no per-bump source review. To size the diff on the
> code you actually review, set `LLAMA_BUMP_EXCLUDE_WEBUI=1` (the helper prints both figures regardless).

### The helper: `.github/scripts/llama-next-version.sh`

It only **reads** — a cached blobless mirror clone of llama.cpp plus `llama/CMakeLists.txt`; it never
edits the repo. It prints the chosen `b<cur> -> b<next>` step, its full and WebUI-excluded diff size,
the commit count, and the `compare` / `.patch` URLs. Environment:

| Var | Default | Meaning |
|---|---|---|
| `LLAMA_BUMP_MAX_DIFF_KB` | `100` | Per-step diff-size threshold, in KiB. |
| `LLAMA_BUMP_EXCLUDE_WEBUI` | `0` | `1` = size the diff **excluding** `tools/ui`. |
| `LLAMA_BUMP_CACHE` | `~/.cache/jllama-llamacpp-mirror` | Mirror-clone location (cloned once, then fetched). |

Worked example — pin `b9859`, latest release `b9866` (full diff 133 KiB ≥ 100 KiB, so it chunks):

```
$ .github/scripts/llama-next-version.sh b9866
current pin    : b9859
latest release : b9866
threshold      : 100 KiB per step (full diff)

next step      : b9859 -> b9862
  diff size    : 45 KiB full  /  ...  KiB excluding tools/ui (auto-followed WebUI)
  commits      : 3
  progress     : intermediate chunk — re-run this script after the bump for the next one
  review diff  : https://github.com/ggml-org/llama.cpp/compare/b9859...b9862
  raw .patch   : https://github.com/ggml-org/llama.cpp/compare/b9859...b9862.patch
```

## 3. Review the chunk's diff

Fetch the printed `compare/...patch` URL (or open the `compare` page). Walk it against the
**priority-ordered API-compatibility review list** in
[`../../CLAUDE.md`](../../CLAUDE.md#files-to-check-for-api-compatibility) — the 8 header rows that have
historically caused breaks (`common.h`, `chat.h`, `speculative.h`, `mtmd.h`, `llama-cpp.h`, `arg.h`,
`llama.h`, `download.h`), plus the project `CMakeLists.txt` for renamed link targets. Note any new
API surface worth wiring through the Java layer (e.g. a new completion param or model-metadata getter).

---

## Applying a bump

Once you have the `b<cur> -> b<next>` step, apply it exactly as
[`CLAUDE.md § Upgrading/Downgrading`](../../CLAUDE.md#upgradingdowngrading-llamacpp-version) describes.
Concretely:

1. **Edit the pin — four files:**
   - `llama/CMakeLists.txt` — the `GIT_TAG b<cur>` line. (It is the only `b<nnnn>` tag in this file — the other two `GIT_TAG` lines pin nlohmann/json `v3.12.0` and GoogleTest `v1.17.0` and must NOT move with a llama.cpp bump. The
     `-DLLAMA_TAG=b<cur>` that once fed the build-time TTS extraction was removed with the
     Qwen3-TTS rework, and the WebUI auto-follows `GIT_TAG` in CI.)
   - `README.md` — the llama.cpp badge and link (version appears twice).
   - `CLAUDE.md` — the "Current llama.cpp pinned version" line (and any build-example `b<nnnn>`).
   - `llama/src/main/java/net/ladenthin/llama/value/LlamaCppVersion.java` — the `LLAMA_CPP_VERSION`
     constant (the pure-Java pin consumers read for a version badge/log line). It mirrors `GIT_TAG`;
     if you forget it, `NativeLibraryLoadSmokeTest.nativeBuildInfoMatchesPinnedVersionConstant` fails
     the build (it cross-checks the constant against `LlamaModel.getLlamaCppBuildInfo()`, which reads
     llama.cpp's own linked-in `build-info`).
2. **Re-verify `patches/`** — a clean configure re-runs the fail-loud `PATCH_COMMAND`, so **every
   `*.patch` in `llama/patches/`** must still apply. Do not maintain a list of them here or anywhere
   else: `apply-llama-patches.cmake` `file(GLOB)`s the directory and applies them in filename order,
   so an enumeration can only go stale (it did, one commit after being written). Use a **fresh**
   build dir: the applier's stamp file pins the patch set to the *checked-out llama.cpp commit*, so
   after a `GIT_TAG` change an existing build dir is exactly the case it refuses to guess at — it
   aborts and tells you to configure fresh, which is what actually re-runs the patches against the
   new source:
   ```bash
   cd llama && mvn -q compile          # generates the OSInfo class CMake's OS-detection needs
   rm -rf build && cmake -B build       # fail-loud: aborts here if any patch no longer applies
   ```
   If a patch no longer applies, refresh its diff against the new source and recommit it.
3. **Check the server contract mechanically when the chunk touches `tools/server/`.** A header diff
   only shows signature changes; it cannot see a *contract* change behind a stable signature. Two
   breaks of that class already shipped — `getMetrics()`'s payload shape (b10408/b10519) and the
   removal of the `-1` = context-size sentinel for `repeat_last_n`/`dry_penalty_last_n` (b10275) —
   and neither was visible to the build. Diff these three sets between the two tags; anything that
   changes has to be traced to the Java layer, not just to the C++ tests:
   ```bash
   # request-field set
   git show b<from>:tools/server/server-schema.cpp | grep -oE 'field_[a-z_]+\("[a-z_0-9]+"' | sort -u
   # request-field bounds
   git show b<from>:tools/server/server-schema.cpp | tr '\n' ' ' \
     | grep -oE 'field_[a-z]+[^(]*\("[a-z_0-9]+"[^;]*?set_(hard_)?limits\([^)]*\)' | sort -u
   # response keys
   # response keys -- BOTH emit forms: brace-init AND res["k"] = ...; the b10585 migration
   # moved `timings`/`prompt_progress` between the two, so a single-form grep reports
   # false removals and would silently miss a new operator[] key.
   git show b<from>:tools/server/server-task.cpp | { grep -oE '\{ *"[A-Za-z_0-9.]+" *,'; \
     git show b<from>:tools/server/server-task.cpp | grep -oE '\[ *"[A-Za-z_0-9.]+" *\] *='; } | sort -u
   ```
   Repeat with `b<to>` and `comm -13` / `comm -23` the two outputs.

4. **Append the history rows** — add a pair of rows to
   [`../history/llama-cpp-breaking-changes.md`](../history/llama-cpp-breaking-changes.md) covering the
   `b<cur> -> b<next>` range (what broke / what was new; "no source change" is a valid row).
5. **Commit + push** on the working branch (do not open a new PR if one already tracks the branch):
   ```bash
   git add llama/CMakeLists.txt README.md CLAUDE.md docs/history/llama-cpp-breaking-changes.md \
           llama/src/main/java/net/ladenthin/llama/value/LlamaCppVersion.java
   git commit -m "Upgrade llama.cpp from b<cur> to b<next>"
   git push -u origin <your-branch>
   ```
6. **Re-run the helper** for the next chunk. Repeat until it reports the **final chunk** (target
   reached).

CI builds every native classifier from the new pin; the full model-backed Java + C++ suites gate the
result. A build failure at the configure step almost always means a patch needs refreshing (step 2).
