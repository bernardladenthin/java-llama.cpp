# Release Process

The maintainer-facing release procedure is **centralized in the workspace repo**:
[`../../workspace/workflows/release-process.md`](../../workspace/workflows/release-process.md).

java-llama.cpp is a **Maven reactor**, so two repo-specific points extend the canonical procedure.

## Reactor version bump (all six poms)

The root `pom.xml` is the parent (`net.ladenthin:llama-parent`); the `llama/`,
`llama-langchain4j/`, `llama-kotlin/`, `llama-ffm/`, and `llama-platform/` modules inherit its version **but hardcode it in their
`<parent><version>`** (there is no `${revision}` single-sourcing). A version change — the release
strip in Step 1 and the post-release bump in Step 3 — must touch **all six poms in lockstep**, or
the reactor build fails with `Could not find artifact net.ladenthin:llama-parent:pom:{VERSION}`.
(The `llama-android/` Gradle build needs no edit — it parses the root pom's version.) Use:

```bash
mvn -q versions:set -DnewVersion={VERSION} -DgenerateBackupPoms=false
```

from the repo root — it updates the root `<version>` plus every child's `<parent><version>` at
once. See the "Version bump" note in [CLAUDE.md](../CLAUDE.md) for the rationale.

**`llama-atmosphere-agent/pom.xml` is not in the reactor**, so `versions:set` skips it: set its
`<version>` to `{VERSION}` in the same commit (`check-natives.py`, in the `code-style` job, fails while
the two differ). It is published at that version right after the reactor deploy.

## Extra README dependency snippet

Besides the root `README.md`, the `llama-langchain4j/README.md` `## Dependency` section, the
`llama-android/README.md` and `llama-kotlin/README.md` Gradle snippets, and the root README's
"Importing in Android" snippets carry **release** dependency versions that must also be set to
`{VERSION}` in Step 1 — they are not covered by the root-README edits and drift silently otherwise
(the release examples stay at `{VERSION}` on the Step 3 snapshot bump).

Also update the dependency examples in `llama-ffm/README.md`, the JBang commands in
`llama-atmosphere-agent/README.md` and `examples/jbang/Chat.java`, and the `<llama.version>` in
`examples/server/pom.xml`. Historical version references describing migrations stay historical.

One reactor `mvn -P release,natives deploy` signs and publishes the parent pom, `llama`,
`llama-langchain4j`, `llama-kotlin`, `llama-ffm`, and `llama-platform` together at the same version,
including the reactor modules' CycloneDX JSON artifacts. CI assembles and checks the complete
native tree before activating `natives`; a local one-platform build must not deploy that profile.
The **Android AARs** (`llama-android`, `llama-android-opencl`) are signed and staged locally
by Gradle in the preparation job. The standalone agent (`net.ladenthin:llama-atmosphere-agent`)
is staged locally too. The Central consumer combines all these frozen outputs into one Portal
bundle. It waits for upload acceptance only, with no asynchronous validation polling.

## GitHub assets and the rolling snapshot

The collection (`java/finalize-publication` of [bernardladenthin/actions](https://github.com/bernardladenthin/actions)) is shared with the three sibling Maven
repositories. It reads `.github/release-assets.json`, discovers the reactor from `pom.xml`, then adds the configured
standalone agent, classifiers in `natives.csv`, Android publications and signed llama.cpp source
archive. The shared interface is documented in
[`../../workspace/policies/release-asset-scripts.md`](../../workspace/policies/release-asset-scripts.md). It copies each artifact and signature, naming `target/bom.json` as
`<artifactId>-<version>-cyclonedx.json`, as Central does. Missing outputs fail the publish job;
incomplete collections are not published. A complete checked package is secured before any
Central transfer, so Central failures cannot hide signed GitHub artifacts.

The `github-snapshot` job serializes the complete upload/update/prune operation.
`java/publish-github` refuses an older commit or an older run/attempt of the same commit.
After a complete upload it moves the `snapshot` tag to that build's commit, updates the release
notes with the commit and workflow run, and removes obsolete assets. An incomplete collection
keeps the existing tag, notes and other assets. The release itself is retained throughout;
versioned releases are never pruned. The first snapshot of a repository is a draft until its upload
is complete.

## Publication gate and upload jobs

The `publish-snapshot` / `publish-release` jobs now prepare locally with `skipUpload=true`,
finish all signing and extra packaging, and verify the complete frozen publication.
The shared `java/finalize-publication` action collects, verifies and secures the
Actions handoff before emitting readiness. The `central-*`, `github-*` and `attest-*` jobs
independently consume that handoff, identically to the three sibling Maven repositories.
Attestation verifies the package and covers all frozen GitHub assets and detached signatures.
None of these consumers rebuilds or signs files. Portal upload acceptance is enough;
validation and automatic publication happen asynchronously. See
[the publication documentation of bernardladenthin/actions](https://github.com/bernardladenthin/actions/blob/main/docs/publication.md).
