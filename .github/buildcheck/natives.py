# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Everything that names a natives jar, checked against .github/natives.csv.

The list is the one place a natives jar is declared. Everything else either reads it (the merge,
the smoke sets) or has to repeat it, and each repetition is checked here:
  * llama/pom.xml       one jar execution per row: classifier, directory, Automatic-Module-Name
  * llama-platform      depends on exactly the rows marked platform=yes
  * publish.yml         a build job uploads natives-<classifier> for every row and no other,
                        `package` waits for each of them (else it packages without that build) and
                        uploads a smoke set llama-smoke-<target> per smoke target, and every smoke
                        target is launched by a job that downloads its set
  * LlamaLoader         LIBRARY_BACKENDS / MODULE_BACKENDS know every backend of their kind (else a
                        jar ships and never loads)
  * CMakeLists.txt      names exactly the backend directories of the list
  * nativedeps.ALLOWED  holds an allowlist for every library directory (cpu, metal), and for no
                        directory the list lacks
  * README.md           documents every classifier
  * examples/           the JBang example and the server example pom name the right jars at the
                        README's version
A row is a `library` (holds libjllama) or a `module` (holds one GPU backend module, additive to the
library jar of its platform). The smoke targets are derived from the list (smoke_targets): every
desktop platform with a library jar, and every module jar's platform must be one of them, so no
natives jar ships unlaunched. package-smoke-sets.sh does not repeat the targets: it asks this
module for them (check-natives.py smoke-targets).

Every check is a function of the texts it compares, so the tests drive them with literals.
"""

import csv
import io
import os
import re
import xml.etree.ElementTree as ET

from . import nativedeps, workflow

NS = {"m": "http://maven.apache.org/POM/4.0.0"}

KINDS = ("library", "module")

# Platforms that get no smoke set: `java -cp` does not apply on Android (the AAR and the emulator
# job do), and no GitHub runner exists for s390x (its C++ suite runs under qemu instead).
SMOKE_EXCLUDED_OSES = ("Linux-Android",)
SMOKE_EXCLUDED_TREES = ("Linux/s390x",)

# `package` uploads each smoke set alone, as llama-smoke-<target>, for the smoke jobs; a matrix row
# is a flow mapping that starts with its target.
SMOKE_ARTIFACT = "llama-smoke-"
MATRIX_ROW = re.compile(r"^\s*-\s*\{\s*target:\s*([a-z0-9-]+)\s*[,}]")
MATRIX_DOWNLOAD = SMOKE_ARTIFACT + "${{ matrix.target }}"


def rows(text):
    """The rows of natives.csv (comment lines and blank lines skipped)."""
    lines = [line for line in io.StringIO(text) if line.strip() and not line.startswith("#")]
    return list(csv.DictReader(lines))


def backend(row):
    return row["directory"].rsplit("/", 1)[1]


def tree(row):
    """<OS>/<ARCH> of the row."""
    return row["directory"].rsplit("/", 1)[0]


def target(row):
    """<os>-<arch> of the row: its classifier without the backend."""
    return row["classifier"][len(backend(row)) + 1:]


def libraries(natives):
    return [r for r in natives if r["kind"] == "library"]


def modules(natives):
    return [r for r in natives if r["kind"] == "module"]


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


def smoke_targets(natives):
    """The smoke targets: the <os>-<arch> of every library row, excluded platforms left out. Sorted.
    A smoke set holds the classes jar, its dependencies and every natives jar of the target, and a
    smoke job launches it on a runner of that OS/arch. package-smoke-sets.sh builds exactly these."""
    return sorted({target(r) for r in libraries(natives)
                   if tree(r).split("/")[0] not in SMOKE_EXCLUDED_OSES and tree(r) not in SMOKE_EXCLUDED_TREES})


def smoke_set(natives, smoke_target):
    """The natives jars of one smoke target: the library row and every module row of its platform."""
    return [r for r in natives if target(r) == smoke_target]


def compare(what, expected, actual):
    return ([f"{what}: missing {name}" for name in sorted(set(expected) - set(actual))]
            + [f"{what}: {name} is not in .github/natives.csv" for name in sorted(set(actual) - set(expected))])


def check_rows(natives):
    failures = []
    classifiers = [r["classifier"] for r in natives]
    if len(set(classifiers)) != len(classifiers):
        failures.append("natives.csv lists a classifier twice")
    trees = {}
    for r in natives:
        if not r["classifier"].startswith(backend(r) + "-") or r["platform"] not in ("yes", "no"):
            failures.append(f"natives.csv: row {r['classifier']} -- classifier must start with its "
                            f"directory's backend '{backend(r)}', platform must be yes or no")
        if r.get("kind") not in KINDS:
            failures.append(f"natives.csv: row {r['classifier']} -- kind must be one of {KINDS}")
        elif r["kind"] == "library" and not re.fullmatch(r"(lib)?jllama\.(so|dll|dylib)", r["library"]):
            failures.append(f"natives.csv: row {r['classifier']} -- a library row must hold the jllama library, "
                            f"not {r['library']}")
        elif r["kind"] == "module":
            if not re.fullmatch(r"(lib)?ggml-[a-z0-9]+\.(so|dll)", r["library"]):
                failures.append(f"natives.csv: row {r['classifier']} -- a module row must hold one ggml backend "
                                f"module (libggml-<x>.so / ggml-<x>.dll), not {r['library']}")
            if r["platform"] == "yes":
                failures.append(f"natives.csv: row {r['classifier']} -- a module cannot be a platform jar: it "
                                f"holds no library")
        trees.setdefault(tree(r), []).append(r)
    for t, group in sorted(trees.items()):
        if not any(r.get("kind") == "library" for r in group):
            failures.append(f"natives.csv: {t} has module jars but no library jar -- nothing would load them")
        names = [r["library"] for r in group]
        if len(set(names)) != len(names):
            failures.append(f"natives.csv: two jars of {t} ship a file of the same name ({names}); they are "
                            f"extracted into one directory")
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
    """The build jobs upload natives-<classifier> for every row, `package` waits for each of them and
    uploads llama-smoke-<target> per smoke target, and every target is downloaded by a smoke job."""
    uploads = {}
    for job in jobs.values():
        for name in job.uploads():
            if name.startswith("natives-"):
                uploads[name[len("natives-"):]] = job.name
    failures = compare("publish.yml natives-* uploads", [r["classifier"] for r in natives], uploads)
    package = jobs.get("package")
    if package is None:
        return failures + ["publish.yml has no `package` job"]
    waited_for = workflow.closure(jobs, "package")
    for classifier, job in sorted(uploads.items()):
        if job not in waited_for:
            failures.append(f"publish.yml: package does not wait for {job}, which uploads natives-{classifier} "
                            f"-- add it to package's needs")
    targets = smoke_targets(natives)
    uploaded = {n[len(SMOKE_ARTIFACT):] for n in package.uploads() if n.startswith(SMOKE_ARTIFACT)}
    failures += compare("publish.yml package smoke-set uploads", targets, uploaded)
    launched = set()
    for job in jobs.values():
        if job.name != "package":
            launched |= smoke_downloads(job)
    failures += compare("publish.yml smoke runs (a job downloading llama-smoke-<target>)", targets, launched)
    for r in modules(natives):
        if tree(r).split("/")[0] not in SMOKE_EXCLUDED_OSES and target(r) not in targets:
            failures.append(f"natives.csv: the module jar {r['classifier']} has no smoke target -- it would ship "
                            f"unlaunched")
    return failures


def smoke_downloads(job):
    """The smoke targets a job downloads: the literal llama-smoke-<target> names in its text, and, when
    it downloads llama-smoke-${{ matrix.target }}, every row of its matrix."""
    launched = set(re.findall(re.escape(SMOKE_ARTIFACT) + r"([a-z0-9]+-[a-z0-9-]+)", job.text))
    launched.discard("${{ matrix.target }}")
    if MATRIX_DOWNLOAD in job.text:
        launched |= {m.group(1) for m in map(MATRIX_ROW.match, job.lines) if m}
    return launched


def check_loader(natives, loader_text):
    """LIBRARY_BACKENDS names every library backend, MODULE_BACKENDS every module backend, and no
    backend is in the wrong list."""
    def constant(name):
        block = re.search(name + r"\s*=(.*?);", loader_text, re.S)
        return set(re.findall(r'"([^"]+)"', block.group(1))) if block else set()
    failures = []
    for kind, const in (("library", "LIBRARY_BACKENDS"), ("module", "MODULE_BACKENDS")):
        listed = {backend(r) for r in natives if r["kind"] == kind}
        known = constant(const)
        failures += [f"LlamaLoader.{const} does not know '{b}' -- its jars would never load"
                     for b in sorted(listed - known)]
        failures += [f"LlamaLoader.{const} names '{b}', which no {kind} jar of natives.csv ships"
                     for b in sorted(known - listed)]
    return failures


def check_cmake(natives, cmake_text):
    named = set(re.findall(r"set\(JLLAMA_BACKEND\s+([A-Za-z0-9_-]+)\)", cmake_text))
    listed = {backend(r) for r in natives}
    return ([f"llama/CMakeLists.txt never sets JLLAMA_BACKEND {b} -- no build writes that directory"
             for b in sorted(listed - named)]
            + [f"llama/CMakeLists.txt sets JLLAMA_BACKEND {b}, which no natives jar ships"
               for b in sorted(named - listed)])


def check_dependency_allowlist(natives, allowed):
    """nativedeps.ALLOWED holds a list for every library directory (the package job would otherwise
    fail on a new one only after every build finished), and none for a directory no jar ships."""
    directories = {r["directory"] for r in natives}
    cpu = {r["directory"] for r in natives if backend(r) in nativedeps.CPU_BACKENDS}
    return ([f"buildcheck/nativedeps.py ALLOWED has no allowlist for {d}" for d in sorted(cpu - set(allowed))]
            + [f"buildcheck/nativedeps.py ALLOWED lists {d}, which no natives jar of natives.csv ships"
               for d in sorted(set(allowed) - directories)])


def check_readme(natives, readme_text):
    return [f"README.md does not document the natives jar `{r['classifier']}`"
            for r in natives if f"`{r['classifier']}`" not in readme_text]


JBANG_EXAMPLE = "examples/jbang/Chat.java"
SERVER_EXAMPLE = "examples/server/pom.xml"


def readme_version(readme_text):
    """The release version of the README's install snippet (llama-platform), which the bump moves by hand."""
    match = re.search(r"<artifactId>llama-platform</artifactId>\s*<version>([^<]+)</version>", readme_text)
    return match.group(1) if match else None


def check_jbang_example(natives, script_text, readme_text):
    """The one-file JBang example runs without a checkout, so it cannot take `llama-platform`: JBang
    treats a `pom` dependency as a BOM and puts nothing of it on the classpath (measured). It therefore
    names the classes jar and the platform=yes natives jars itself -- exactly those, at the version of
    the README's install snippet."""
    deps = [line.split()[1] for line in script_text.splitlines() if line.startswith("//DEPS ")]
    coords = [d.split(":") for d in deps]
    good = [c for c in coords if c[:2] == ["net.ladenthin", "llama"] and len(c) in (3, 4)]
    failures = [f"{JBANG_EXAMPLE}: `{d}` is not net.ladenthin:llama:<version>[:<classifier>]"
                for d, c in zip(deps, coords) if c not in good]
    expected = readme_version(readme_text)
    versions = {c[2] for c in good}
    if versions != {expected}:
        failures.append(f"{JBANG_EXAMPLE}: DEPS version(s) {sorted(versions)} must be the README install "
                        f"snippet's {expected}")
    if not any(len(c) == 3 for c in good):
        failures.append(f"{JBANG_EXAMPLE}: the classes jar net.ladenthin:llama:<version> is missing")
    named = [c[3] for c in good if len(c) == 4]
    platform = [r["classifier"] for r in natives if r["platform"] == "yes"]
    failures += [f"{JBANG_EXAMPLE}: missing the platform natives jar {c}" for c in platform if c not in named]
    failures += [f"{JBANG_EXAMPLE}: names {c}, which is not a platform=yes row of .github/natives.csv"
                 for c in named if c not in platform]
    if len(set(named)) != len(named):
        failures.append(f"{JBANG_EXAMPLE}: a natives jar is named twice")
    return failures


def check_server_example(natives, pom_text, readme_text):
    """examples/server/pom.xml starts the server from Maven Central: llama-platform at the README's
    version, plus one profile per GPU module jar (id = the classifier) that adds exactly that jar, so
    `-P cuda13-linux-x86-64` is the whole opt-in. Checked both ways: a profile per module row, no
    profile for a jar that does not exist."""
    root = ET.fromstring(pom_text)
    version = root.findtext("m:properties/m:llama.version", namespaces=NS)
    expected = readme_version(readme_text)
    failures = []
    if version != expected:
        failures.append(f"{SERVER_EXAMPLE}: <llama.version> is {version}, the README install snippet's is {expected}")
    profiles = {}
    for profile in root.iterfind("m:profiles/m:profile", NS):
        pid = profile.findtext("m:id", namespaces=NS)
        deps = profile.findall("m:dependencies/m:dependency", NS)
        classifiers = [d.findtext("m:classifier", namespaces=NS) for d in deps]
        if pid in {r["classifier"] for r in modules(natives)} or (classifiers and classifiers[0] in
                                                                     {r["classifier"] for r in natives}):
            profiles[pid] = classifiers
    failures += compare(f"{SERVER_EXAMPLE} GPU profiles", [r["classifier"] for r in modules(natives)], profiles)
    for pid, classifiers in sorted(profiles.items()):
        if classifiers != [pid]:
            failures.append(f"{SERVER_EXAMPLE}: profile {pid} must add exactly the natives jar {pid}, it adds "
                            f"{classifiers}")
    return failures


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
    readme = read(root, "README.md")
    return (check_rows(natives)
            + check_pom(natives, read(root, "llama/pom.xml"))
            + check_platform(natives, read(root, "llama-platform/pom.xml"))
            + check_workflow(natives, workflow.parse(read(root, ".github/workflows/publish.yml")))
            + check_loader(natives, read(root, "llama/src/main/java/net/ladenthin/llama/loader/LlamaLoader.java"))
            + check_cmake(natives, read(root, "llama/CMakeLists.txt"))
            + check_dependency_allowlist(natives, nativedeps.ALLOWED)
            + check_readme(natives, readme)
            + check_jbang_example(natives, read(root, JBANG_EXAMPLE), readme)
            + check_server_example(natives, read(root, SERVER_EXAMPLE), readme)
            + check_agent_version(read(root, "pom.xml"), read(root, "llama-atmosphere-agent/pom.xml")))
