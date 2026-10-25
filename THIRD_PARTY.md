<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Third-party license and attribution notes

The core is MIT. JMH/JCStress, Lincheck, JUnit and jqwik are test dependencies.
Logback is used by test tools; the core runtime binder is SLF4J Simple.
The standalone atmosphere agent has additional Jetty EE10 runtime dependencies.
Native llama.cpp artifacts have their own third-party contents and require separate review.

## Review evidence

See [the dependency license review](docs/dependency-license-review.md) in the source repository
for exact-version evidence and remaining decisions. This inventory is not an approval of all
licenses found by a scanner. Test-tool redistribution requires a separate review.

For unchanged Maven components, source is available from
`https://repo.maven.apache.org/maven2/<groupId with slashes>/<artifactId>/<version>/`
as `<artifactId>-<version>-sources.jar`; POMs and upstream source repositories provide provenance.
Preserve copyright, license and NOTICE files when redistributing dependencies.

java-llama.cpp ships no fat JAR. Each published JAR carries its own classes (a natives JAR its
native libraries) and, under `META-INF/third-party/net.ladenthin/<artifactId>/`, this file and
`LICENSE`; Maven resolves every dependency as an artifact of its own, with its own legal files.
`.github/collect-third-party-licenses.py`, which collects dependency legal files into the runnable
fat JARs of BitcoinAddressFinder and srcmorph, is kept here byte-identical with them but not run.
This source-location statement does not clear unknown proprietary or historical Git findings.
