# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Collect the artifacts published by this repository, preserving partial output on failure.

The reactor's POMs declare the modules; natives.csv declares its classifiers. Every reactor
module also attaches a CycloneDX BOM as target/bom.json, whose Central filename is different.
Missing artifacts/signatures fail the collection, but available files are still attached by CI.
"""

import csv
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def projects(root):
    """Root and recursively declared reactor modules, followed by the standalone agent."""
    root = Path(root).resolve()
    seen = set()
    result = []

    def visit(folder, reactor=True):
        folder = folder.resolve()
        if not folder.is_relative_to(root) or folder in seen:
            raise ValueError(f"invalid or repeated Maven module: {folder}")
        seen.add(folder)
        pom = ET.parse(folder / "pom.xml").getroot()
        artifact = pom.findtext("{*}artifactId")
        version = pom.findtext("{*}version") or pom.findtext("{*}parent/{*}version")
        packaging = pom.findtext("{*}packaging", "jar")
        if not artifact or not version or "${" in version or packaging not in ("pom", "jar"):
            raise ValueError(f"unsupported release coordinates in {folder / 'pom.xml'}")
        if result and version != result[0][2]:
            raise ValueError(f"release version mismatch in {folder / 'pom.xml'}: {version}")
        result.append((folder, artifact, version, packaging, reactor))
        for module in pom.findall("{*}modules/{*}module"):
            visit(folder / module.text.strip())

    visit(root)
    visit(root / "llama-atmosphere-agent", reactor=False)
    return result


def collect(root, destination):
    """Return missing/invalid outputs after copying every available artifact and signature."""
    root, destination = Path(root), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError(f"asset destination must be empty: {destination}")
    modules = projects(root)
    version = modules[0][2]
    expected = []
    errors = []
    for folder, artifact, _, packaging, reactor in modules:
        base = f"{artifact}-{version}"
        expected.append((folder / "target" / f"{base}.pom", f"{base}.pom"))
        if packaging == "jar":
            for classifier in ("", "-sources", "-javadoc"):
                name = f"{base}{classifier}.jar"
                expected.append((folder / "target" / name, name))
        if reactor:
            expected.append((folder / "target/bom.json", f"{base}-cyclonedx.json"))

    with (root / ".github/natives.csv").open() as stream:
        for row in csv.DictReader(line for line in stream if line.strip() and not line.startswith("#")):
            name = f"llama-{version}-{row['classifier']}.jar"
            expected.append((root / "llama/target" / name, name))

    for artifact, publication in (("llama-android", "llamaAndroid"),
                                  ("llama-android-opencl", "llamaAndroidOpencl")):
        base = f"{artifact}-{version}"
        android = root / "llama-android/build"
        expected.extend(((android / "aar" / f"{base}.aar", f"{base}.aar"),
                         (android / "publications" / publication / "pom-default.xml", f"{base}.pom")))
        for classifier in ("sources", "javadoc"):
            expected.append((android / "distributions" / f"llama-android-{version}-{classifier}.jar",
                             f"{base}-{classifier}.jar"))

    archives = list((root / "llama-source-archive").glob(f"llama.cpp-*-jllama-{version}-src.tar.gz"))
    if len(archives) != 1:
        errors.append(f"expected one signed llama.cpp source archive for {version}, found {len(archives)}")
    else:
        expected.append((archives[0], archives[0].name))

    names = set()
    for source, name in expected:
        if name in names:
            raise ValueError(f"duplicate release asset filename: {name}")
        names.add(name)
        for suffix in ("", ".asc"):
            path = Path(str(source) + suffix)
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"missing or empty release output: {path}")
                continue
            shutil.copyfile(path, destination / (name + suffix))
    return errors
