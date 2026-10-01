# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Everything that names a natives jar, checked against .github/natives.csv.

The list is the one place a natives jar is declared. Everything else either reads it (the merge,
the fat-jar assembly) or has to repeat it, and each repetition is checked here:
  * llama/pom.xml       one jar execution per row: classifier, directory, Automatic-Module-Name
  * llama-platform      depends on exactly the rows marked platform=yes
  * publish.yml         a build job uploads natives-<classifier> for every row and no other, and
                        `package` waits for each of them (else it packages without that build)
  * LlamaLoader         BACKEND_PRIORITY tries every backend (else a jar ships and never loads)
  * CMakeLists.txt      names exactly the backend directories of the list
  * nativedeps.ALLOWED  holds an allowlist for every CPU directory (cpu, metal, msvc), and for no
                        directory the list lacks
  * README.md           documents every classifier
and, for the all-backends fat jars derived from the list (fatjar_targets), that each one is
uploaded as llama-fatjar-smoke-<target>, launched by a smoke job (a script line naming it, or a row
of the smoke-fatjar matrix), named in the agent jar's Class-Path and in the README.
package-fatjars.sh does not repeat the targets at all: it asks this module for them.

Every check is a function of the texts it compares, so the tests drive them with literals.
"""

import csv
import io
import os
import re
import xml.etree.ElementTree as ET

from . import nativedeps, workflow

NS = {"m": "http://maven.apache.org/POM/4.0.0"}

# Backends that get no place in an all-backends jar: msvc is the same CPU build from another
# generator, so it would only be a second CPU library the loader tries before the first.
FATJAR_EXCLUDED_BACKENDS = ("msvc",)
# Platforms that get no all-backends jar: `java -jar` does not apply on Android (the AAR does).
FATJAR_EXCLUDED_OSES = ("Linux-Android",)

FATJAR_NAME = re.compile(r"all-([a-z0-9]+(?:-[a-z0-9]+)*?)-jar-with-dependencies")

# package-fatjars uploads each all-backends fat jar alone, as llama-fatjar-smoke-<target>, for the
# smoke-fatjar matrix; a matrix row is a flow mapping that starts with its target.
SMOKE_ARTIFACT = "llama-fatjar-smoke-"
MATRIX_ROW = re.compile(r"^\s*-\s*\{\s*target:\s*([a-z0-9-]+)\s*[,}]")
MATRIX_FATJAR = "all-${{ matrix.target }}-jar-with-dependencies"


def rows(text):
    """The rows of natives.csv (comment lines and blank lines skipped)."""
    lines = [line for line in io.StringIO(text) if line.strip() and not line.startswith("#")]
    return list(csv.DictReader(lines))


def backend(row):
    return row["directory"].rsplit("/", 1)[1]


def tree(row):
    """<OS>/<ARCH> of the row."""
    return row["directory"].rsplit("/", 1)[0]


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


def fatjar_targets(natives):
    """The all-backends fat jars: one per <os>-<arch> (the classifier suffix) with more than one
    natives jar, excluded backends and platforms left out. Sorted. package-fatjars.sh builds
    exactly these (it asks for them: check-natives.py fatjar-targets)."""
    groups = {}
    for row in natives:
        if backend(row) in FATJAR_EXCLUDED_BACKENDS or tree(row).split("/")[0] in FATJAR_EXCLUDED_OSES:
            continue
        target = row["classifier"][len(backend(row)) + 1:]
        groups.setdefault(target, set()).add(backend(row))
    return sorted(t for t, backends in groups.items() if len(backends) > 1)


def fatjar_names(text):
    """The fat-jar targets a text names (`...-all-<target>-jar-with-dependencies...`)."""
    return set(FATJAR_NAME.findall(text))


def compare(what, expected, actual):
    return ([f"{what}: missing {name}" for name in sorted(set(expected) - set(actual))]
            + [f"{what}: {name} is not in .github/natives.csv" for name in sorted(set(actual) - set(expected))])


def check_rows(natives):
    failures = []
    classifiers = [r["classifier"] for r in natives]
    if len(set(classifiers)) != len(classifiers):
        failures.append("natives.csv lists a classifier twice")
    for r in natives:
        if not r["classifier"].startswith(backend(r) + "-") or r["platform"] not in ("yes", "no"):
            failures.append(f"natives.csv: row {r['classifier']} -- classifier must start with its "
                            f"directory's backend '{backend(r)}', platform must be yes or no")
    return failures


def check_pom(natives, pom_text):
    """The natives profile of llama/pom.xml: one jar execution per row."""
    found = {}
    for profile in ET.fromstring(pom_text).iterfind("m:profiles/m:profile", NS):
        if profile.findtext("m:id", namespaces=NS) != "natives":
            continue
        for ex in profile.iterfind(".//m:plugin/m:executions/m:execution", NS):
            conf = ex.find("m:configuration", NS)
            classifier = conf.findtext("m:classifier", namespaces=NS) if conf is not None else None
            if classifier:
                found[classifier] = (conf.findtext("m:includes/m:include", namespaces=NS),
                                     conf.findtext("m:archive/m:manifestEntries/m:Automatic-Module-Name",
                                                   namespaces=NS))
    by_classifier = {r["classifier"]: r for r in natives}
    failures = compare("llama/pom.xml natives profile", by_classifier, found)
    for classifier in sorted(set(found) & set(by_classifier)):
        want = (f"net/ladenthin/llama/{by_classifier[classifier]['directory']}/**", module_name(classifier))
        if found[classifier] != want:
            failures.append(f"llama/pom.xml: {classifier} has {found[classifier]}, expected {want} "
                            f"(check-natives.py pom prints the executions)")
    return failures


def check_platform(natives, platform_pom_text):
    deps = {d.findtext("m:classifier", namespaces=NS)
            for d in ET.fromstring(platform_pom_text).iterfind("m:dependencies/m:dependency", NS)} - {None}
    return compare("llama-platform/pom.xml", [r["classifier"] for r in natives if r["platform"] == "yes"], deps)


def check_workflow(natives, jobs):
    """The build jobs upload natives-<classifier> for every row, `package` waits for each of them,
    and every fat-jar target is uploaded for a smoke job and launched by one."""
    uploads = {}
    for job in jobs.values():
        for name in job.uploads():
            if name.startswith("natives-"):
                uploads[name[len("natives-"):]] = job.name
    failures = compare("publish.yml natives-* uploads", [r["classifier"] for r in natives], uploads)
    if "package" not in jobs:
        return failures + ["publish.yml has no `package` job"]
    waited_for = workflow.closure(jobs, "package")
    for classifier, job in sorted(uploads.items()):
        if job not in waited_for:
            failures.append(f"publish.yml: package does not wait for {job}, which uploads natives-{classifier} "
                            f"-- add it to package's needs")
    targets = fatjar_targets(natives)
    assembler = jobs.get("package-fatjars")
    if assembler is None:
        return failures + ["publish.yml has no `package-fatjars` job"]
    failures += check_smoke_uploads(targets, assembler)
    launched = set()
    for job in jobs.values():
        if job.name != "package-fatjars":
            launched |= smoke_runs(job, failures)
    failures += compare("publish.yml fat-jar smoke runs", targets, launched)
    return failures


def check_smoke_uploads(targets, assembler):
    """package-fatjars uploads llama-fatjar-smoke-<target> for every target, each holding that
    target's jar (the path names the same target as the artifact)."""
    failures, uploaded = [], set()
    for step in assembler.steps():
        text = "\n".join(step)
        names = [n for n in re.findall(r"name:\s*(\S+)", text) if n.startswith(SMOKE_ARTIFACT)]
        if "actions/upload-artifact@" not in text or not names:
            continue
        target = names[0][len(SMOKE_ARTIFACT):]
        uploaded.add(target)
        if fatjar_names("\n".join(line for line in step if "path:" in line)) != {target}:
            failures.append(f"publish.yml package-fatjars: {names[0]} does not upload the {target} fat jar")
    return compare("publish.yml package-fatjars smoke-jar uploads", targets, uploaded) + failures


def smoke_runs(job, failures):
    """The fat-jar targets a job launches: the ones its smoke-script lines name, and, when those run
    `all-${{ matrix.target }}-...`, every row of its matrix -- which must then download each row's
    own smoke jar."""
    smoke = "\n".join(line for line in job.lines if ".github/smoke-" in line)
    if not smoke:
        return set()
    launched = fatjar_names(smoke)
    if MATRIX_FATJAR in job.text:
        launched |= {m.group(1) for m in map(MATRIX_ROW.match, job.lines) if m}
        if f"name: {SMOKE_ARTIFACT}${{{{ matrix.target }}}}" not in job.text:
            failures.append(f"publish.yml: {job.name} does not download {SMOKE_ARTIFACT}${{{{ matrix.target }}}}")
    return launched


def check_loader(natives, loader_text):
    block = re.search(r"BACKEND_PRIORITY\s*=(.*?);", loader_text, re.S)
    priority = set(re.findall(r'"([^"]+)"', block.group(1))) if block else set()
    return [f"LlamaLoader.BACKEND_PRIORITY does not try '{b}' -- its jars would never load"
            for b in sorted({backend(r) for r in natives} - priority)]


def check_cmake(natives, cmake_text):
    named = set(re.findall(r"set\(JLLAMA_BACKEND\s+([A-Za-z0-9_-]+)\)", cmake_text))
    listed = {backend(r) for r in natives}
    return ([f"llama/CMakeLists.txt never sets JLLAMA_BACKEND {b} -- no build writes that directory"
             for b in sorted(listed - named)]
            + [f"llama/CMakeLists.txt sets JLLAMA_BACKEND {b}, which no natives jar ships"
               for b in sorted(named - listed)])


def check_dependency_allowlist(natives, allowed):
    """nativedeps.ALLOWED holds a list for every CPU directory (the package job would otherwise fail
    on a new one only after every build finished), and none for a directory no jar ships."""
    directories = {r["directory"] for r in natives}
    cpu = {r["directory"] for r in natives if backend(r) in nativedeps.CPU_BACKENDS}
    return ([f"buildcheck/nativedeps.py ALLOWED has no allowlist for {d}" for d in sorted(cpu - set(allowed))]
            + [f"buildcheck/nativedeps.py ALLOWED lists {d}, which no natives jar of natives.csv ships"
               for d in sorted(set(allowed) - directories)])


def check_readme(natives, readme_text):
    failures = [f"README.md does not document the natives jar `{r['classifier']}`"
                for r in natives if f"`{r['classifier']}`" not in readme_text]
    return failures + compare("README.md all-backends fat jars", fatjar_targets(natives), fatjar_names(readme_text))


def check_agent_class_path(natives, agent_pom_text):
    """`java -jar` on the agent jar finds the core through its manifest Class-Path, which names
    every all-backends fat jar."""
    match = re.search(r"<Class-Path>(.*?)</Class-Path>", agent_pom_text, re.S)
    named = fatjar_names(match.group(1)) if match else set()
    return compare("llama-atmosphere-agent/pom.xml Class-Path", fatjar_targets(natives), named)


def check_agent_version(root_pom_text, agent_pom_text):
    """The agent is published to Maven Central at the core's version and depends on the core of its
    own version, so a release that bumps the reactor without it would publish an agent naming a
    core that does not exist (or an old one). It is not a reactor module, so `versions:set` misses it."""
    def version(text):
        element = ET.fromstring(text).find("m:version", NS)
        return None if element is None else (element.text or "").strip()
    core, agent = version(root_pom_text), version(agent_pom_text)
    if core == agent:
        return []
    return [f"llama-atmosphere-agent/pom.xml is version {agent}, the reactor {core}: set both to the same version"]


def read(root, path):
    with open(os.path.join(root, path), encoding="utf-8") as f:
        return f.read()


def check(root):
    """Every check, over the files of the repository at `root`."""
    natives = rows(read(root, ".github/natives.csv"))
    return (check_rows(natives)
            + check_pom(natives, read(root, "llama/pom.xml"))
            + check_platform(natives, read(root, "llama-platform/pom.xml"))
            + check_workflow(natives, workflow.parse(read(root, ".github/workflows/publish.yml")))
            + check_loader(natives, read(root, "llama/src/main/java/net/ladenthin/llama/loader/LlamaLoader.java"))
            + check_cmake(natives, read(root, "llama/CMakeLists.txt"))
            + check_dependency_allowlist(natives, nativedeps.ALLOWED)
            + check_readme(natives, read(root, "README.md"))
            + check_agent_class_path(natives, read(root, "llama-atmosphere-agent/pom.xml"))
            + check_agent_version(read(root, "pom.xml"), read(root, "llama-atmosphere-agent/pom.xml")))
