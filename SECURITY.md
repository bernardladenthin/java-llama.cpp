# Security Policy

## Supported versions

Security fixes target the most recent release of the `6.x` series. The `5.x` series was the first to be published under the `net.ladenthin:llama` Maven coordinates; releases `1.x` through `4.2.0` were authored by the pre-fork upstream (`kherud/java-llama.cpp`) and are not maintained here.

| Version | Supported |
|---------|-----------|
| 6.x (latest published on Maven Central) | Yes |
| 5.x | No (upgrade to 6.x, see the CHANGELOG's "Upgrading from 5.1") |
| < 5.0 | No (see upstream `kherud/java-llama.cpp`) |

## Reporting a vulnerability — primary channel

Please use GitHub's private vulnerability reporting form:

<https://github.com/bernardladenthin/java-llama.cpp/security/advisories/new>

This channel is private and visible only to repository maintainers.

## Reporting a vulnerability — fallback

If you cannot use the GitHub advisory form, email the maintainer:

**bernard.ladenthin@gmail.com**

PGP / signed email is not required but is welcome; the public key can be requested in your first message.

## Response SLA

- Acknowledgement: within **14 days** of receipt.
- Initial remediation timeline (fix ETA or rejection rationale): within **30 days** of receipt.

## Coordinated disclosure

Default embargo: **90 days** from the date the report is acknowledged, or until a fix is published and a GitHub Security Advisory is issued — whichever comes first. The embargo may be shortened or extended by mutual agreement with the reporter; please keep details private until the embargo lifts.

## Scope

**In scope:**

- The Java API surface (`net.ladenthin.llama.*`).
- The project's native layer under `llama/src/main/cpp/`: the JNI shells (`jllama.cpp`, `native_server.cpp`, `rpc_bridge.cpp`), the C API (`jllama_api.h`/`.cpp`) and the engine behind both (`jllama_engine.cpp`, the helper headers), and the llama.cpp patches this project carries (`llama/patches/`).
- The project's native build glue (`llama/CMakeLists.txt`, `llama/cmake/`, `.github/build*.sh`, dockcross scripts) — including issues that result in unsafe linkage, missing hardening flags, or supply-chain risk in our build pipeline.
- Maven Central artefact integrity (signing, publishing workflow).

**Out of scope:**

- Vulnerabilities in upstream [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) (the native engine) or its dependencies (GGML backends, cpp-httplib, etc.). Please report those upstream; if the upstream fix requires a corresponding update here, file a follow-up issue once the upstream advisory is published.
- Vulnerabilities in third-party model weights or training data.
- Vulnerabilities in user-supplied prompts or model configuration.

## Security update notifications

- **Watch** the repository (Watch &gt; Custom &gt; Security alerts) to receive GitHub Security Advisories as they are published.
- **Subscribe to Releases** to be notified when a new Maven Central artefact is published with security fixes.
