# Release Process

The maintainer-facing release procedure is **centralized in the workspace repo**:
[`../workspace/workflows/release-process.md`](../workspace/workflows/release-process.md).

java-llama.cpp is a **Maven reactor**, so two repo-specific points extend the canonical procedure.

## Reactor version bump (all four poms)

The root `pom.xml` is the parent (`net.ladenthin:llama-parent`); the `llama/`,
`llama-langchain4j/`, and `llama-kotlin/` modules inherit its version **but hardcode it in their
`<parent><version>`** (there is no `${revision}` single-sourcing). A version change — the release
strip in Step 1 and the post-release bump in Step 3 — must touch **all four poms in lockstep**, or
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

One reactor `mvn -P release deploy` signs and publishes the parent pom, `llama`,
`llama-langchain4j`, and `llama-kotlin` together at the same version. The **Android AARs**
(`llama-android`, `llama-android-opencl`) are published by the `publish-release` job's separate
Gradle step (signed Central Portal bundle upload) — no manual action, but they appear as their own
deployment named `llama-android-{VERSION}` in the Central Portal UI. The agent
(`net.ladenthin:llama-atmosphere-agent`) is likewise its own deployment, from the step after the
reactor deploy.
