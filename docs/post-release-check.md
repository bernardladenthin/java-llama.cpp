<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT
-->

# Post-release check

What to verify **after** a release has been published, from the outside: against the GitHub
release, Maven Central and a real JVM, not against the build tree it came from.

This is not the release procedure — that is [`RELEASE.md`](RELEASE.md) and the canonical
[workspace release procedure](../../workspace/workflows/release-process.md). Those describe how a
release is *made*; this file describes how a release is *audited once it exists*. The two are kept
apart on purpose: a green pipeline is evidence about the build, and every failure class recorded in
`CLAUDE.md` that reached a user reached them *through* a green pipeline — a jar that loaded on the
runner and nowhere else, a dylib assembled from two artifacts, a model-gated suite that reported
nothing and passed. So every check below is phrased to be answerable **without trusting the run that
produced the artifacts**, and several deliberately recompute from the tagged sources what the
artifacts merely claim.

**Run it on the published artifacts, every release.** A check that can only say "CI was green" does
not belong in this file.

## Conventions

Each check states **what it proves** (the failure class it would catch), **how** to run it, and the
**pass criterion**. Nothing is hardcoded to a version: the expected sets are derived from the
repository at the tag — above all [`.github/natives.csv`](../.github/natives.csv), the one list every
packaging step reads (`CLAUDE.md`, "Natives jars"). Throughout:

```bash
VERSION=<the released version>      # e.g. 6.0.0
TAG="v$VERSION"
RUN=<the publishing run id>         # the workflow_dispatch run on $TAG
GH=bernardladenthin/java-llama.cpp
CENTRAL=https://repo1.maven.org/maven2/net/ladenthin
```

Work in a scratch directory, and keep the downloaded artifacts in a directory of their own — they
are untrusted input until checked, so no script is run from inside it.

```bash
gh release download "$TAG" --dir assets
```

The whole publication is a few hundred MB (the two CUDA module jars dominate). Download it once:
every check below reads those bytes, and Central is proven equal to them in check 6 rather than
fetched a second time.

## 1. The ref, the run and the tag agree

**What it proves.** That the artifacts came from the commit the tag names, through the publishing
workflow, and not from a re-pointed tag or a second attempt whose inputs differed. A tag is a moving
reference; during release preparation it is routinely deleted and re-created, so a cancelled run on
an earlier commit is normal and the only safe statement is the one tying *tag → commit → run →
artifact* together.

```bash
git fetch origin --tags --force
git rev-parse "$TAG^{commit}"
gh run list --workflow publish.yml --limit 10 \
  --json databaseId,headBranch,headSha,event,conclusion,runAttempt
```

**Pass.** Exactly one successful `workflow_dispatch` run on `$TAG`; its `headSha` equals the tag
commit; the tag commit is an ancestor of `main`. Note the attempt number — a re-run attempt is not
wrong, but every later check must then name that same attempt.

## 2. Every job either succeeded or was skipped for a stated reason

**What it proves.** That nothing red hid behind a release gate, and that the only skipped jobs are
the ones the other channel owns. A skip is not benign: GitHub propagates a skip down the whole
`needs:` chain past any intervening `if:`, which once turned a run "successful" while `package`,
every smoke set and both publish jobs did nothing (`CLAUDE.md`, "Code signing").

```bash
gh run view "$RUN" --json jobs \
  --jq '.jobs | group_by(.conclusion) | map({c:.[0].conclusion, n:length})'
gh run view "$RUN" --json jobs --jq '.jobs[] | select(.conclusion!="success") | .conclusion+" "+.name'
```

**Pass.** No failure and no cancellation. The skipped set is exactly the *other* channel's jobs — on
a release run, the snapshot channel check, its gate, its Central upload, its GitHub pre-release and
its attestation. Any other skip is a finding, not a detail.

## 3. The build was cache-free

**What it proves.** That the shared compiler cache is outside the release's trust chain. A `v*` tag
build must be from scratch whatever the dispatch input said (`CLAUDE.md`, "CI build cache"), so no
shipped object can come out of a third-party cache.

```bash
gh api "repos/$GH/actions/jobs/<a native build job id>/logs" | grep -iE 'USE_CACHE|sccache (ON|OFF)'
```

**Pass.** `USE_CACHE: false` in the job environment **and** the build script's own line reporting the
launcher off. Both matter: the first is the intent, the second is what the build did — the build
scripts self-enable a cache they find on `PATH` unless explicitly told not to, which has silently
re-enabled one before.

## 4. The publication is complete, and its size is derived rather than remembered

**What it proves.** That no artifact is missing and none is unexpected. The number to compare against
is computed from the repository, never copied from the last release: the natives rows of
`natives.csv`, the reactor modules, the standalone agent and the two Android publications.

```bash
gh release view "$TAG" --json assets --jq '.assets[] | "\(.size)\t\(.name)"'
```

Every file carries a detached `.asc`, so the asset count is even and exactly twice the file count.
Derive the expected Central set: for the core, the classes jar + sources + javadoc + pom + CycloneDX
plus one jar per `natives.csv` row; for each other reactor module its own four or five; for the agent
and the two AARs theirs. The GitHub set is that plus the **signed llama.cpp source archive**, which is
a GitHub-only file of the publication (`CLAUDE.md`, "Source archive").

**Pass.** The derived Central count equals the count the publication gate reported ("Publication
ready: … N Central artifacts"), and the GitHub file count is that plus one. Two independent routes to
the same number; agreeing on a wrong number requires both to be wrong the same way.

## 5. Presence on Maven Central

**What it proves.** That what was uploaded is actually resolvable. The Portal accepts a bundle and
validates asynchronously, so the gate's success is not evidence that the artifacts are published.

```bash
for c in $(grep -v '^#' .github/natives.csv | tail -n +2 | cut -d, -f1); do
  curl -s -o /dev/null -w "%{http_code} $c\n" "$CENTRAL/llama/$VERSION/llama-$VERSION-$c.jar"
done
```

**Pass.** HTTP 200 for every natives jar and its `.asc`, for every module's jar/sources/javadoc/pom,
for both AARs and for every pom-only module. A classifier that 404s is invisible to CI, because CI
never resolves from Central.

## 6. Every signature verifies, from one key

**What it proves.** That the publication is signed by the release key throughout, not partly. The
publication gate verifies signatures as well, but with the key the runner imported; this re-verifies
with the key a consumer would trust, and additionally pins that **one** signing subkey signed
everything.

```bash
cd assets
for a in *.asc; do gpg --status-fd 1 --verify "$a" "${a%.asc}" | grep -oE 'VALIDSIG [A-F0-9]+'; done \
  | awk '{print $2}' | sort | uniq -c
gpg --list-keys --with-subkey-fingerprints <release key fingerprint>
```

**Pass.** `GOODSIG` for every file; a single `VALIDSIG` fingerprint across all of them; and that
fingerprint is the `[S]` subkey of the release key. A mixed set of signing keys is a finding even when
every signature is individually good.

## 7. Build provenance covers every file

**What it proves.** That each file was produced by this workflow at this commit — which the `.asc`
cannot say, since it attests the key and not the build (`CLAUDE.md`, "Modular natives"). Checking a
sample is not enough: the question is whether anything is *un*attested.

```bash
gh attestation verify --owner bernardladenthin --format json "assets/llama-$VERSION.jar" > att.json
```

Then, from the attestations' subject lists, confirm every local file appears **and** its SHA-256
matches the recorded digest, and report any local file that appears in neither list.

**Pass.** Each attestation names `refs/tags/$TAG`, the tag commit, `publish.yml` and the run (and
attempt) from check 1; every published file is covered with a matching digest; nothing is unattested.
Expect more than one attestation and more subjects than files — the `package` job attests the
individual native libraries too — so a subject that is not a release file is normal, while a release
file that is not a subject is not.

## 8. The natives jars match the list

**What it proves.** That each jar is the one `natives.csv` declares and carries its provenance
record. This is the merge step's check re-run on the *published* jar: the same properties, read from
the shipped bytes rather than from the build tree.

Per row of `natives.csv`, assert the jar holds exactly one directory
`net/ladenthin/llama/<directory>/` (plus `META-INF/`), contains the declared library, carries
`jllama-build.txt` with **all** provenance keys — `llama.cpp`, `llama.cpp.commit`, `backend`, `kind`,
`classifier`, `jllama.version`, `jllama.commit`, `patches`, `patches.sha256`, `compiler`, `toolkit`,
`signed` — whose `classifier`, `kind` and `jllama.version` equal the row and the release, has
`jllama-sbom.json` beside it, and declares
`Automatic-Module-Name: net.ladenthin.llama.natives.<classifier with _ for ->`. Unfold the manifest's
72-byte line wraps before reading that name: a wrapped value looks truncated and is not. A `module`
row holds exactly four files — the module, the two list files and the BOM.

**Pass.** Zero disagreements over all rows. Also compare the per-directory file counts with the ones
`CLAUDE.md` documents per platform ("Modular natives"): a plausible-looking count that moved is how a
dropped CPU variant would present.

## 9. The provenance is internally consistent, and the patch set is recomputed

**What it proves.** That the binaries were built from the tagged sources — and this is the one check
that takes the artifact's word for nothing. `patches.sha256` is a hash over `<name> <sha256>\n` per
patch; recomputing it from the patches committed at the tag ties the shipped binaries to the reviewed
patch set. Reading the key out of the jar proves only that the jar is self-consistent.

```bash
git ls-tree --name-only "$TAG" llama/patches/ | sort          # extract each with git show
for f in *.patch; do printf '%s %s\n' "$f" "$(sha256sum "$f" | cut -d' ' -f1)"; done | sha256sum
```

**Pass.** One `llama.cpp` tag and one `llama.cpp.commit` across all natives jars; `jllama.version`
the release; `jllama.commit` the tag commit **with no `-dirty` suffix**; and the recomputed hash equal
to the `patches.sha256` every jar declares. Record each GPU jar's `toolkit` as well — it is the vendor
SDK that module was built against, and nothing else reports it.

## 10. The bytecode floors

**What it proves.** That the artifact a consumer loads can be loaded by the JVM the README promises.
The core's floor is Java 8 (major 52), held by two deliberate pom decisions that are easy to undo by
accident (`CLAUDE.md`, "Java 8 bytecode floor"). The other modules have higher floors on purpose, so
a single global threshold would be wrong.

Scan every class of every jar, skipping `module-info.class` and `META-INF/versions/**` (a classpath
JVM loads neither):

| Artifact | Expected class-file major |
|---|---|
| the classes jar and every natives jar | 52, plus a single `module-info` at 53 |
| `llama-kotlin` | 52 |
| `llama-langchain4j` | 61 (Java 17) |
| `llama-ffm` | 66 (Java 22) |
| the standalone agent | 65 (Java 21) |
| each AAR's `classes.jar` | 52, and the classes jar's count minus `module-info` |

**Pass.** No class above its artifact's floor — and an empty scan is a failure rather than a pass: a
directory a failed build left empty is the reason that rule exists.

## 11. A consumer can resolve it

**What it proves.** That the published poms resolve from Central for a real build, property
substitutions included — those come only from the parent pom. CI never resolves the release from
Central, so nothing upstream of this check has tried it.

Write a throwaway consumer pom depending on `llama-platform` (`<type>pom</type>`) plus a GPU
classifier or two, and resolve it against an **empty local repository**:

```bash
mvn -B -Dmaven.repo.local=./m2 dependency:tree
mvn -B -Dmaven.repo.local=./m2 dependency:copy-dependencies -DoutputDirectory=./cp -DincludeScope=runtime
```

**Pass.** `llama-platform` brings the classes jar and the library jar of every `platform=yes` row; the
classes jar brings jspecify, Jackson and `slf4j-api` at compile and `slf4j-simple` at runtime; and the
graph contains **neither logback nor checker-qual** — both would break a Java 8 consumer, and both are
kept out by pom scope alone, which no compile-time check can see.

## 12. It runs — on the oldest supported JDK

**What it proves.** The thing nothing else proves. CI runs the jars on its runners; this runs the
*published* jars, resolved from Central, on the floor JDK. And a load is not the check: in the Windows
measurements recorded in `CLAUDE.md` every crash came *after* a successful load that reported the
right device count, so the criterion is real inference, not initialization.

On the classpath from check 11, compile a small driver with the **floor JDK's** `javac` against the
published jars and run it on that JDK. Assert, in order: the provenance map (`getJllamaBuildInfo()`)
matches check 9; `getLlamaCppBuildInfo()` starts with the `LlamaCppVersion` constant — the pure-Java
pin and the linked binary must agree; a streamed generation yields tokens; a tokenize/detokenize round
trip; a blocking completion. Then read the loader's own line from stderr.

**Pass.** Every assertion holds on the floor JDK, with no `UnsupportedClassVersionError` and no SLF4J
provider complaint (that is check 11's pom scopes, observed at runtime). The loader line names the
library backend **and every GPU module jar on the classpath**, and ggml's registry line names the
backends and devices it ended up with — on a GPU host the accelerator must appear, which is the only
evidence that a shipped GPU module opened against a real driver instead of failing silently. Record
the throughput; it is the baseline the next release is read against.

## 13. The source archive stands on its own

**What it proves.** That the release stays rebuildable without upstream. The archive is the pinned
upstream commit plus `llama/patches/`, and because `FetchContent` skips population for an overridden
source directory, the applier never runs for a consumer rebuilding from it — the stamp is what proves
the patches are in (`CLAUDE.md`, "Source archive").

```bash
tar -xzf "assets/llama.cpp-<tag>-jllama-$VERSION-src.tar.gz"
cat <dir>/.jllama-patches-applied
```

**Pass.** The stamp's `head` equals the `llama.cpp.commit` the natives jars declare; its per-patch
hashes equal the tag's patches line for line (the same lines check 9 recomputed); and each patch's
distinctive added symbol is actually present in the unpacked tree — the stamp records intent, a grep
for what each patch adds records the result. The pack is reproducible: a single mtime (the commit's
time), numeric owner `0/0`, no owner names. Entry order is GNU tar's per-directory `--sort=name`, so
it is deterministic without being flat lexicographic — do not read that as unsorted.

## 14. The Android artifacts

**What it proves.** The two invariants that make the AARs consumable, each of which has failed before:
Google Play's 16 KB page size, and a `DT_NEEDED` set bionic can satisfy.

**Pass.** Both AARs declare the documented `minSdkVersion`; the CPU AAR's `classes.jar` is the classes
jar minus `module-info` and it carries every ABI's library set under `jni/<abi>/`; the additive AAR
holds exactly its one module, an **empty** `classes.jar` (duplicating the API breaks D8) and a pom
depending on the CPU AAR at `<type>aar</type>` and the same version; both carry their consumer
ProGuard rules; and **every** `LOAD` segment of **every** released Android ELF — in the AARs and in
the Android natives jars alike — is aligned to at least 16384.

## 15. The server surface and the embedded WebUI

**What it proves.** That the headline consumer-facing surface works from the published jars: the full
upstream server inside `libjllama`, with the WebUI assets compiled in. A local `cmake` build uses the
empty-asset stub, so "the WebUI is served" is a property of the *released* jar specifically and of no
build a developer makes.

Start `ServerLauncher` from the check-11 classpath with a model, then probe it.

**Pass.** `/health` answers 200; `GET /` answers 200 with `Content-Type: text/html` and
`Content-Encoding: gzip`, and the body decompresses to the app shell rather than to an error page;
`/props` answers; and a `/v1/chat/completions` request returns a choice whose `system_fingerprint` is
the linked build from check 9. An empty-asset stub answers `/` with 404 — that is the single
observation separating a released jar from a local one.

## 16. Standing gaps to re-read every release

Not failures — properties of the current release that a reader should be told rather than left to
discover. Each is observed by a check above and is expected to read a certain way until the work
behind it lands.

- **`signed=no` in every natives jar.** The Windows DLLs are not Authenticode-signed; the signing path
  exists and stays inert until the certificate and its four secrets do (`CLAUDE.md`, "Code signing").
  The compensating controls are the detached `.asc` and the Sigstore attestation of check 7. Re-read
  `signed=` each release: once a release signs, the value flips and
  `verify-native-deps.py --require-signed` becomes the gate.
- **The next development version: bump the _minor_, not the patch.** After a release, `main` moves to
  the next **minor** `-SNAPSHOT` (`6.0.0` → `6.1.0-SNAPSHOT`), deliberately leaving the patch number
  free so an emergency release off the released line can always take it (`6.0.1`). The bump covers all
  six reactor poms **and** the standalone agent pom — `versions:set` reaches the reactor but not the
  agent, so that one is edited by hand (`CLAUDE.md`, "Repository layout").

  Until the bump lands, `main` still names the released version, and the next snapshot publication is
  refused by its own channel check — correctly, since that channel only publishes a `-SNAPSHOT`, but
  the refusal reads like a workflow defect. Right after a release the bump is commonly still in flight
  on a branch, which is normal; what this check asks is that it is not *forgotten*. So read it off
  `main` and, when it is not there yet, name the branch that carries it:

  ```bash
  git show origin/main:pom.xml | grep -m1 -A1 llama-parent
  git branch -r --contains <the bump commit>
  ```
- **Scope drift between the reactor poms and the Gradle-built AAR poms.** The AAR poms take their
  dependency *versions* from the core pom, but their *scopes* are the Gradle build's own. Compare them
  every release: a dependency the core deliberately keeps out of the consumer graph at `provided` must
  not reappear at `compile` in an AAR pom.
