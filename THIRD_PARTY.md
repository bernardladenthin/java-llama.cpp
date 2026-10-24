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

Runnable BAF/srcmorph CLI packaging uses Python 3.8+ to collect original dependency
legal files before the project JAR and fat JAR are assembled. Override `-Dlicense.python=python`
if that is the Python command on the build host. Legal files are under
`META-INF/third-party/dependencies/<original-jar-name>-<content-hash>/`.
`inventory.json` maps exact JAR hashes/classifiers to preserved files and also lists JARs
without embedded legal material, which require review or supplemental notices.
This source-location statement does not clear unknown proprietary or historical Git findings.
