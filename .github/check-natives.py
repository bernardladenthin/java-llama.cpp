#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Fail when anything that names the natives jars disagrees with .github/natives.csv.

The list is the one place a natives jar is declared. Everything else either reads it (the merge,
the fat-jar assembly) or must repeat it, and is checked here:
  * llama/pom.xml       one jar execution per row: classifier, directory, Automatic-Module-Name
  * llama-platform      depends on exactly the rows marked platform=yes
  * publish.yml         a build job uploads natives-<classifier> for every row, and no other
  * LlamaLoader         BACKEND_PRIORITY tries every backend (else a jar ships and never loads)

Usage:
  check-natives.py          check, exit 1 on any disagreement
  check-natives.py pom      print the natives jar executions for llama/pom.xml
"""

import csv
import os
import re
import sys
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NS = {"m": "http://maven.apache.org/POM/4.0.0"}


def rows():
    with open(os.path.join(ROOT, ".github", "natives.csv"), encoding="utf-8") as f:
        lines = [line for line in f if line.strip() and not line.startswith("#")]
    return list(csv.DictReader(lines))


def module_name(classifier):
    """Each natives jar needs its own: without one, every jar derives the module name `llama`
    from its file name, and the module path silently keeps only the first."""
    return "net.ladenthin.llama.natives." + classifier.replace("-", "_")


def pom_execution(row):
    return f"""\t\t\t\t\t\t\t<execution>
\t\t\t\t\t\t\t\t<id>natives-{row['classifier']}</id>
\t\t\t\t\t\t\t\t<phase>package</phase>
\t\t\t\t\t\t\t\t<goals>
\t\t\t\t\t\t\t\t\t<goal>jar</goal>
\t\t\t\t\t\t\t\t</goals>
\t\t\t\t\t\t\t\t<configuration>
\t\t\t\t\t\t\t\t\t<classifier>{row['classifier']}</classifier>
\t\t\t\t\t\t\t\t\t<classesDirectory>${{project.basedir}}/src/main/natives</classesDirectory>
\t\t\t\t\t\t\t\t\t<includes>
\t\t\t\t\t\t\t\t\t\t<include>net/ladenthin/llama/{row['directory']}/**</include>
\t\t\t\t\t\t\t\t\t</includes>
\t\t\t\t\t\t\t\t\t<archive>
\t\t\t\t\t\t\t\t\t\t<addMavenDescriptor>false</addMavenDescriptor>
\t\t\t\t\t\t\t\t\t\t<manifestEntries>
\t\t\t\t\t\t\t\t\t\t\t<Automatic-Module-Name>{module_name(row['classifier'])}</Automatic-Module-Name>
\t\t\t\t\t\t\t\t\t\t</manifestEntries>
\t\t\t\t\t\t\t\t\t</archive>
\t\t\t\t\t\t\t\t</configuration>
\t\t\t\t\t\t\t</execution>"""


def pom_natives(path):
    """classifier -> (include, module name) for every jar execution of the natives profile."""
    tree = ET.parse(path)
    found = {}
    for profile in tree.getroot().iterfind("m:profiles/m:profile", NS):
        if profile.findtext("m:id", namespaces=NS) != "natives":
            continue
        for ex in profile.iterfind(".//m:plugin/m:executions/m:execution", NS):
            conf = ex.find("m:configuration", NS)
            classifier = conf.findtext("m:classifier", namespaces=NS) if conf is not None else None
            if classifier:
                found[classifier] = (conf.findtext("m:includes/m:include", namespaces=NS),
                                     conf.findtext("m:archive/m:manifestEntries/m:Automatic-Module-Name",
                                                   namespaces=NS))
    return found


def compare(what, expected, actual, failures):
    for name in sorted(expected - actual):
        failures.append(f"{what}: missing {name}")
    for name in sorted(actual - expected):
        failures.append(f"{what}: {name} is not in .github/natives.csv")


def main(argv):
    natives = rows()
    if len(argv) > 1 and argv[1] == "pom":
        print("\n".join(pom_execution(r) for r in natives))
        return 0
    failures = []
    by_classifier = {r["classifier"]: r for r in natives}
    if len(by_classifier) != len(natives):
        failures.append("natives.csv lists a classifier twice")
    for r in natives:
        backend, rest = r["directory"].rsplit("/", 1)[1], r["classifier"]
        if not rest.startswith(backend + "-") or r["platform"] not in ("yes", "no"):
            failures.append(f"natives.csv: row {r['classifier']} -- classifier must start with its "
                            f"directory's backend '{backend}', platform must be yes or no")

    pom = pom_natives(os.path.join(ROOT, "llama", "pom.xml"))
    compare("llama/pom.xml natives profile", set(by_classifier), set(pom), failures)
    for classifier in sorted(set(pom) & set(by_classifier)):
        want = (f"net/ladenthin/llama/{by_classifier[classifier]['directory']}/**", module_name(classifier))
        if pom[classifier] != want:
            failures.append(f"llama/pom.xml: {classifier} has {pom[classifier]}, expected {want} "
                            f"(check-natives.py pom prints the executions)")

    platform = ET.parse(os.path.join(ROOT, "llama-platform", "pom.xml"))
    deps = {d.findtext("m:classifier", namespaces=NS)
            for d in platform.getroot().iterfind("m:dependencies/m:dependency", NS)} - {None}
    compare("llama-platform/pom.xml", {r["classifier"] for r in natives if r["platform"] == "yes"}, deps, failures)

    with open(os.path.join(ROOT, ".github", "workflows", "publish.yml"), encoding="utf-8") as f:
        uploads = set(re.findall(r"upload-artifact@\S+\s+with:\s+name:\s*natives-(\S+)", f.read()))
    compare("publish.yml natives-* uploads", set(by_classifier), uploads, failures)

    loader = os.path.join(ROOT, "llama", "src", "main", "java", "net", "ladenthin", "llama", "loader",
                          "LlamaLoader.java")
    with open(loader, encoding="utf-8") as f:
        block = re.search(r"BACKEND_PRIORITY\s*=(.*?);", f.read(), re.S)
    priority = set(re.findall(r'"([^"]+)"', block.group(1))) if block else set()
    for backend in sorted({r["directory"].rsplit("/", 1)[1] for r in natives} - priority):
        failures.append(f"LlamaLoader.BACKEND_PRIORITY does not try '{backend}' -- its jars would never load")

    for f in failures:
        print(f"::error::{f}", file=sys.stderr)
    print(f"{len(natives)} natives jars, {len(failures)} disagreements")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
