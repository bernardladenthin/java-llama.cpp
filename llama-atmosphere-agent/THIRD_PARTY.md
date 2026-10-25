<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Third-party license and attribution notes: llama-atmosphere-agent

The agent is MIT (`LICENSE` next to this file, the same text as the repository's). It ships as a
thin jar; Maven resolves every runtime dependency as an artifact of its own, with its own legal
files:

| Dependency | License |
|---|---|
| `net.ladenthin:llama` and the natives jars `llama-platform` names | MIT; their own `THIRD_PARTY.md` and the notices of the compiled llama.cpp code travel in those jars |
| Atmosphere (`org.atmosphere:atmosphere-ai`) | Apache-2.0 |
| Jetty EE10 (`org.eclipse.jetty.ee10:*`) | EPL-2.0 OR Apache-2.0 |
| JLine (`org.jline:jline`) | BSD-3-Clause |
| Agent Client Protocol SDK (`com.agentclientprotocol:acp-*`) | Apache-2.0 |
| Jackson, JSpecify | Apache-2.0 |
| SLF4J Simple | MIT |

The repository's `THIRD_PARTY.md` and [the dependency license review](https://github.com/bernardladenthin/actions/blob/main/docs/dependency-license-review.md) of bernardladenthin/actions hold the review
evidence.
This folder is meant to be copied out and run on its own, which is why it carries both files
itself; `check-natives.py` holds `LICENSE` byte-identical to the repository's.
