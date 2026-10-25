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
                        jar ships and never loads), and OPT_IN_BACKENDS names exactly the backends
                        the optin column declares (else an opt-in jar takes over silently, or an
                        inert one has no row saying why)
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

from . import nativedeps

NS = {"m": "http://maven.apache.org/POM/4.0.0"}

KINDS = ("library", "module")

# Platforms that get no smoke set: `java -cp` does not apply on Android (the AAR and the emulator
# job do), and no GitHub runner exists for s390x (its C++ suite runs under qemu instead, and
# smoke-glibc-floor loads its jar under qemu on RHEL 8 -- a load, not the model-backed server smoke).
SMOKE_EXCLUDED_OSES = ("Linux-Android",)
# The job that re-uploads the Windows natives artifacts Authenticode-signed (CLAUDE.md, "Code signing").
SIGN_JOB = "sign-windows-natives"
# A job-level key sits at the four-space indent of the job body (`    if:`); a step's `if:` is deeper.
JOB_CONDITION = re.compile(r"^    if:")
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


# Where the natives profile copies llama/legal/natives/ before packaging (CMake writes the natives
# tree next to it, so it is a generated directory); every natives jar includes it.
LEGAL_INCLUDE = "META-INF/third-party/natives/**"


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
\t\t\t\t\t\t\t\t\t\t<include>{LEGAL_INCLUDE}</include>
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


# The glibc-floor smoke (publish.yml smoke-glibc-floor): every library jar whose directory promises a
# glibc floor (nativedeps.GLIBC_CEILING) is loaded on the oldest distribution of that floor, RHEL 8
# (glibc 2.28 and GCC 8's libstdc++.so.6), on its own architecture -- s390x under qemu. verify-native-
# deps.py reads the symbol versions; only a load on the old runtime proves the jar starts there, and
# none ran before the s390x jar of 6.0.0 shipped needing glibc 2.38 (CLAUDE.md, "Linux s390x"). `package`
# uploads one set (the classes jar, its dependencies and these jars) as FLOOR_ARTIFACT, and the job has
# one matrix row per floor target, which check_floor holds to the list.
FLOOR_ARTIFACT = "llama-glibc-floor"
FLOOR_JOB = "smoke-glibc-floor"


def floor_set(natives):
    """The library rows whose directory promises a glibc floor (nativedeps.GLIBC_CEILING), in list order.
    Module rows are left out: a GPU module needs its vendor runtime, which an RHEL 8 image does not have."""
    return [r for r in libraries(natives) if r["directory"] in nativedeps.GLIBC_CEILING]


def floor_targets(natives):
    """The <os>-<arch> of every floor_set row, sorted: one smoke-glibc-floor matrix row each."""
    return sorted(target(r) for r in floor_set(natives))


def check_floor(natives, jobs):
    """`package` uploads the floor set as FLOOR_ARTIFACT, and FLOOR_JOB downloads it and launches one
    matrix row per floor target: a library jar that promises a glibc floor does not ship unloaded on it."""
    targets = floor_targets(natives)
    if not targets:
        return []
    failures = []
    package = jobs.get("package")
    if package is not None and FLOOR_ARTIFACT not in package.uploads():
        failures.append(f"publish.yml: package does not upload {FLOOR_ARTIFACT}, the set {FLOOR_JOB} loads "
                        f"(package-smoke-sets.sh writes it)")
    job = jobs.get(FLOOR_JOB)
    if job is None:
        return failures + [f"publish.yml has no `{FLOOR_JOB}` job, which loads {', '.join(targets)} on RHEL 8"]
    downloads = any(any("actions/download-artifact@" in line for line in step)
                    and any(line.strip() == f"name: {FLOOR_ARTIFACT}" for line in step) for step in job.steps())
    if not downloads:
        failures.append(f"publish.yml {FLOOR_JOB} does not download {FLOOR_ARTIFACT}")
    launched = {m.group(1) for m in map(MATRIX_ROW.match, job.lines) if m}
    failures += [f"publish.yml {FLOOR_JOB}: no matrix row for {t}, whose jar promises the glibc floor of "
                 f"nativedeps.GLIBC_CEILING" for t in sorted(set(targets) - launched)]
    failures += [f"publish.yml {FLOOR_JOB}: the matrix row {t} is no library jar with a glibc floor "
                 f"(nativedeps.GLIBC_CEILING)" for t in sorted(launched - set(targets))]
    return failures


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
        if r.get("optin") not in ("yes", "no"):
            failures.append(f"natives.csv: row {r['classifier']} -- optin must be yes or no")
        elif r["optin"] == "yes" and r.get("kind") != "module":
            failures.append(f"natives.csv: row {r['classifier']} -- only a module can be opt-in: a library jar "
                            f"is the one jar that must load")
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
    for b in sorted({backend(r) for r in natives}):
        answers = {r.get("optin") for r in natives if backend(r) == b}
        if len(answers) > 1:
            failures.append(f"natives.csv: backend '{b}' is opt-in in one row and not in another "
                            f"({sorted(answers)}) -- the loader decides per backend, so the rows cannot differ")
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
                found[classifier] = (tuple(i.text for i in conf.iterfind("m:includes/m:include", NS)),
                                     conf.findtext("m:archive/m:manifestEntries/m:Automatic-Module-Name",
                                                   namespaces=NS))
    by_classifier = {r["classifier"]: r for r in natives}
    failures = compare("llama/pom.xml natives profile", by_classifier, found)
    for classifier in sorted(set(found) & set(by_classifier)):
        want = ((f"net/ladenthin/llama/{by_classifier[classifier]['directory']}/**", LEGAL_INCLUDE),
                module_name(classifier))
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
    from actionlib.common import workflow  # here, not at the top: smoke-targets/smoke-set need no actionlib
    uploads = {}
    for job in jobs.values():
        for name in job.uploads():
            if name.startswith("natives-"):
                uploads.setdefault(name[len("natives-"):], set()).add(job.name)
    failures = compare("publish.yml natives-* uploads", [r["classifier"] for r in natives], uploads)
    package = jobs.get("package")
    if package is None:
        return failures + ["publish.yml has no `package` job"]
    waited_for = workflow.closure(jobs, "package")
    for classifier, uploaders in sorted(uploads.items()):
        for job in sorted(uploaders):
            if job not in waited_for:
                failures.append(f"publish.yml: package does not wait for {job}, which uploads natives-{classifier} "
                                f"-- add it to package's needs")
    failures += check_signing(natives, jobs, uploads)
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


def check_signing(natives, jobs, uploads):
    """`sign-windows-natives` re-uploads every Windows natives artifact signed: one upload-artifact step
    per Windows row of the list, with `overwrite: true` (the same name as the build job's artifact) and
    `if-no-files-found: error` (a download that matched fewer artifacts must not leave the unsigned
    build artifact in place), nothing of another platform, and the job waits for every job that uploads
    or reads one of them -- `overwrite` deletes the artifact for a moment, so a build or test job still
    reading it would race (the Windows Java test job reads natives-cpu-windows-x86-64 through the
    reusable workflow's input). A Windows row without its step would ship that jar unsigned on a
    signed run, which `--require-signed` in `package` then reports late.

    And the job carries no job-level `if:`: it must run on every run (a no-op without the switch,
    gated step by step) rather than be skipped, because GitHub applies a skip to every job in the
    dependency chain behind the skipped one, past any condition of the jobs in between -- the first
    form was `if:`-skipped without the switch, `package` tolerated that with `!cancelled()` and ran
    green, and the smoke sets, the report and both publish jobs behind `package` were skipped with
    the run concluding "success" (run 38870889665, attempt 2)."""
    from actionlib.common import workflow  # here, not at the top: smoke-targets/smoke-set need no actionlib
    windows = [r["classifier"] for r in natives if tree(r).split("/")[0] == "Windows"]
    sign = jobs.get(SIGN_JOB)
    if sign is None:
        return [f"publish.yml has no `{SIGN_JOB}` job"] if windows else []
    signed = {c for c, uploaders in uploads.items() if SIGN_JOB in uploaders}
    failures = [f"publish.yml {SIGN_JOB} re-uploads (one per Windows row): missing {c}"
                for c in sorted(set(windows) - signed)]
    if any(JOB_CONDITION.match(line) for line in sign.lines):
        failures.append(f"publish.yml {SIGN_JOB} carries a job-level `if:` -- the job must run on every run "
                        f"(a no-op without the switch), never be skipped: a skipped job skips every job behind "
                        f"`package` in the dependency chain, past any condition in between")
    failures += [f"publish.yml {SIGN_JOB} re-uploads natives-{c}, which is not a Windows row -- it would replace "
                 f"the build artifact of another platform with an unsigned copy"
                 for c in sorted(signed - set(windows))]
    uploaders = set()
    for classifier in sorted(signed & set(windows)):
        for job in sorted(uploads[classifier] - {SIGN_JOB}):
            uploaders.add(job)
            if job not in sign.needs:
                failures.append(f"publish.yml: {SIGN_JOB} does not wait for {job}, which uploads "
                                f"natives-{classifier} -- add it to the job's needs")
    downstream = {name for name in jobs if SIGN_JOB in workflow.closure(jobs, name)}
    for name, job in jobs.items():
        if name == SIGN_JOB or name in downstream or name in uploaders or name in sign.needs:
            continue
        code = "\n".join(line for line in job.lines if not line.lstrip().startswith("#"))
        read = [c for c in windows if f"natives-{c}" in code]
        if read:
            failures.append(f"publish.yml: {name} reads natives-{read[0]} but {SIGN_JOB} does not wait for it -- the "
                            f"signed re-upload deletes the artifact for a moment; add {name} to the job's needs")
    for step in sign.steps():
        if not any("actions/upload-artifact@" in line for line in step):
            continue
        if not any(line.strip() == "overwrite: true" for line in step):
            failures.append(f"publish.yml {SIGN_JOB}: an upload-artifact step without `overwrite: true` would fail "
                            f"on the artifact the build job uploaded")
        if not any(line.strip() == "if-no-files-found: error" for line in step):
            failures.append(f"publish.yml {SIGN_JOB}: an upload-artifact step without `if-no-files-found: error` "
                            f"would leave the unsigned build artifact in place when the download matched nothing")
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
    # OPT_IN_BACKENDS is what makes a module jar inert until the property names it, so a backend the
    # list declares opt-in while the loader does not would ship a jar that takes over silently --
    # which is the measured OpenVINO regression this column exists for -- and the other way round a
    # jar would never load with no row saying why.
    listed_optin = {backend(r) for r in natives if r.get("optin") == "yes"}
    known_optin = constant("OPT_IN_BACKENDS")
    failures += [f"LlamaLoader.OPT_IN_BACKENDS does not know '{b}', which natives.csv declares opt-in"
                 for b in sorted(listed_optin - known_optin)]
    failures += [f"LlamaLoader.OPT_IN_BACKENDS names '{b}', which no row of natives.csv declares opt-in"
                 for b in sorted(known_optin - listed_optin)]
    for kind, const in (("library", "LIBRARY_BACKENDS"), ("module", "MODULE_BACKENDS")):
        listed = {backend(r) for r in natives if r["kind"] == kind}
        known = constant(const)
        failures += [f"LlamaLoader.{const} does not know '{b}' -- its jars would never load"
                     for b in sorted(listed - known)]
        failures += [f"LlamaLoader.{const} names '{b}', which no {kind} jar of natives.csv ships"
                     for b in sorted(known - listed)]
    # MODULE_LIBRARY_NAMES is how the directory branch (net.ladenthin.llama.lib.path) recognises a
    # module that lies flat next to the library: backend -> base library name, which must be the
    # `library` column of every module row with its lib prefix and extension removed, in both
    # directions. (A module backend has one base name on every OS: libggml-cuda.so / ggml-cuda.dll.)
    expected = {}
    for r in natives:
        if r["kind"] == "module":
            base = re.sub(r"^lib", "", re.sub(r"\.(so|dll|dylib)$", "", r["library"]))
            expected.setdefault(backend(r), set()).add(base)
    block = re.search(r"names\.put\(.*?MODULE_LIBRARY_NAMES = ", loader_text, re.S)
    declared = dict(re.findall(r'names\.put\("([^"]+)",\s*"([^"]+)"\)', block.group(0))) if block else {}
    for b, bases in sorted(expected.items()):
        if len(bases) != 1:
            failures.append(f"natives.csv: the module backend '{b}' ships differently named libraries: {sorted(bases)}")
        elif declared.get(b) != next(iter(bases)):
            failures.append(f"LlamaLoader.MODULE_LIBRARY_NAMES maps '{b}' to '{declared.get(b)}', natives.csv ships "
                            f"'{next(iter(bases))}' -- a module in a lib.path directory would go unrecognised")
    failures += [f"LlamaLoader.MODULE_LIBRARY_NAMES names '{b}', which no module jar of natives.csv ships"
                 for b in sorted(set(declared) - set(expected))]
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


# The READMEs whose dependency snippets name a release version by hand (CLAUDE.md "Version bump"); the
# langchain4j one is the snippet that was forgotten on the 5.0.4 -> 5.0.5 bump, the kotlin one on 5.1.0 -> 6.0.0.
MODULE_READMES = ("llama-langchain4j/README.md", "llama-kotlin/README.md", "llama-android/README.md",
                  "llama-ffm/README.md")
SNIPPET_VERSION = re.compile(r"<groupId>net\.ladenthin</groupId>\s*<artifactId>[a-z0-9-]+</artifactId>\s*"
                             r"<version>([^<]+)</version>|[\"']net\.ladenthin:[a-z0-9-]+:([^\"':]+)(?::[a-z0-9-]+)?[\"']")


def snippet_versions(text):
    """Every release version a Maven (`<groupId>net.ladenthin</groupId> ... <version>`) or Gradle
    (`"net.ladenthin:<artifact>:<version>"`) dependency snippet of `text` names, in order."""
    return [a or b for a, b in SNIPPET_VERSION.findall(text)]


def check_module_readmes(readme_text, module_readmes):
    """The dependency snippets of the module READMEs (`module_readmes`: path -> text) and the root README's
    own `llama-langchain4j` snippet name the version of the root README's install snippet -- the bump
    edits every one of them by hand, and a forgotten one points consumers at the previous release."""
    expected = readme_version(readme_text)
    failures = []
    for path, text in sorted(module_readmes.items()):
        failures += [f"{path}: a dependency snippet names {v}, the README install snippet's version is {expected}"
                     for v in snippet_versions(text) if v != expected]
    failures += [f"README.md: the llama-langchain4j snippet names {v}, the install snippet {expected}"
                 for v in re.findall(r"<artifactId>llama-langchain4j</artifactId>\s*<version>([^<]+)</version>",
                                     readme_text) if v != expected]
    return failures


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


def check_agent_license(root_license_text, agent_license_text):
    """The agent folder carries its own LICENSE (it is copied out and run on its own) and its jar
    ships that copy, so it must stay the repository's text."""
    if root_license_text == agent_license_text:
        return []
    return ["llama-atmosphere-agent/LICENSE differs from LICENSE: copy the repository's LICENSE there"]


def read(root, path):
    with open(os.path.join(root, path), encoding="utf-8") as f:
        return f.read()


def check(root):
    """Every check, over the files of the repository at `root`."""
    from actionlib.common import workflow  # here, not at the top: smoke-targets/smoke-set need no actionlib
    natives = rows(read(root, ".github/natives.csv"))
    readme = read(root, "README.md")
    return (check_rows(natives)
            + check_pom(natives, read(root, "llama/pom.xml"))
            + check_platform(natives, read(root, "llama-platform/pom.xml"))
            + check_workflow(natives, workflow.parse(read(root, ".github/workflows/publish.yml")))
            + check_floor(natives, workflow.parse(read(root, ".github/workflows/publish.yml")))
            + check_loader(natives, read(root, "llama/src/main/java/net/ladenthin/llama/loader/LlamaLoader.java"))
            + check_cmake(natives, read(root, "llama/CMakeLists.txt"))
            + check_dependency_allowlist(natives, nativedeps.ALLOWED)
            + check_readme(natives, readme)
            + check_jbang_example(natives, read(root, JBANG_EXAMPLE), readme)
            + check_server_example(natives, read(root, SERVER_EXAMPLE), readme)
            + check_module_readmes(readme, {p: read(root, p) for p in MODULE_READMES})
            + check_agent_version(read(root, "pom.xml"), read(root, "llama-atmosphere-agent/pom.xml"))
            + check_agent_license(read(root, "LICENSE"), read(root, "llama-atmosphere-agent/LICENSE")))
