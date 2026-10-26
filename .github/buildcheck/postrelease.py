# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""The post-release checks of this repository's own artifacts, on the published release files.

docs/post-release-check.md lists what to verify once a release exists. The checks every repository
of bernardladenthin/actions shares (the run and its jobs, the files against the contract, Maven
Central, the signatures, the provenance attestation, the bytecode floors, the next version) are its
`java/audit-publication` action; this module holds the ones only this repository has. It reads the
files that action downloaded and the checkout of the tag, and takes the artifacts' word for nothing
it can recompute:

  natives   (check 8)  every natives jar is the one natives.csv declares: one directory, its library,
                       jllama-build.txt with every provenance key and the row's classifier and kind,
                       jllama-sbom.json, the Automatic-Module-Name; a module jar holds its module,
                       the two lists and the BOM only
  provenance (check 9) one llama.cpp tag and commit across all jars, the release version, the tag's
                       commit without -dirty, and patches / patches.sha256 RECOMPUTED from the
                       patches at the tag -- the one check that ties the binaries to the reviewed
                       patch set rather than to themselves
  source    (check 13) the source archive's stamp names the jars' llama.cpp commit and the tag's
                       patches line for line, an added line of every patch is in the unpacked tree,
                       and the pack is reproducible (one mtime, owner 0/0, no owner names)
  android   (check 14) the AAR invariants: minSdk, the CPU AAR's classes.jar is the classes jar minus
                       module-info and its jni/ libraries are the Android natives jars' bytes, the
                       OpenCL AAR holds its one module and an empty classes.jar, both carry consumer
                       rules, both poms mirror the core (aarpoms.py) and the OpenCL one names the
                       CPU AAR of this version, and every Android ELF is 16 KB aligned
  cache     (check 3)  the tag build ran without the shared compiler cache: no job log shows
                       USE_CACHE true or a build script reporting sccache ON, and some show it off

Usage (the CLI is .github/post-release-check.py): post-release-check.py --tag v<version>
--assets <dir> [--run <id>] [--checks natives,provenance,...]. Standard library only.
"""

import argparse
import hashlib
import io
import os
import re
import subprocess
import sys
import tarfile
import zipfile

from . import aarpoms, natives, nativedeps

CHECKS = ("natives", "provenance", "source", "android", "cache")
BUILD_KEYS = ("llama.cpp", "llama.cpp.commit", "backend", "kind", "classifier", "jllama.version",
              "jllama.commit", "patches", "patches.sha256", "compiler", "toolkit", "signed")
MODULE_FILES = {"jllama-build.txt", "jllama-files.txt", "jllama-sbom.json"}
ANDROID_ABIS = {"arm64-v8a": "cpu-android-aarch64", "x86_64": "cpu-android-x86-64"}
MIN_SDK = "28"


class Result:
    def __init__(self, name):
        self.name, self.problems, self.notes = name, [], []

    def need(self, condition, message):
        if not condition:
            self.problems.append(message)
        return condition


def read_rows(root):
    with open(os.path.join(root, ".github", "natives.csv"), encoding="utf-8") as stream:
        return natives.rows(stream.read())


def _files(archive):
    return [n for n in archive.namelist() if not n.endswith("/")]


def build_info(text):
    """The key=value lines of a jllama-build.txt (comments skipped)."""
    out = {}
    for line in text.replace("\r", "").splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            out[key] = value
    return out


def manifest_attribute(text, name):
    """One main-section attribute of a MANIFEST.MF, its 72-byte continuation lines unfolded."""
    unfolded = re.sub(r"\r?\n ", "", text)
    for line in unfolded.splitlines():
        if line.startswith(name + ":"):
            return line[len(name) + 1:].strip()
    return None


# --- natives ------------------------------------------------------------------------------------

def check_natives(assets, rows, version):
    """(result, builds): builds maps each classifier to its jllama-build.txt keys."""
    result = Result("natives")
    builds = {}
    for row in rows:
        classifier, directory, library, kind = row["classifier"], row["directory"], row["library"], row["kind"]
        path = os.path.join(assets, f"llama-{version}-{classifier}.jar")
        if not result.need(os.path.isfile(path), f"{classifier}: llama-{version}-{classifier}.jar is not in the release"):
            continue
        with zipfile.ZipFile(path) as jar:
            prefix = f"net/ladenthin/llama/{directory}/"
            files = _files(jar)
            stray = [f for f in files if not f.startswith(prefix) and not f.startswith("META-INF/")]
            result.need(not stray, f"{classifier}: files outside {prefix}: {stray[:3]}")
            inside = {f[len(prefix):] for f in files if f.startswith(prefix)}
            result.need(library in inside, f"{classifier}: {prefix}{library} is missing")
            result.need("jllama-sbom.json" in inside, f"{classifier}: no jllama-sbom.json")
            if kind == "module":
                result.need(inside == MODULE_FILES | {library},
                            f"{classifier}: a module jar holds {library} and the lists only, found {sorted(inside)}")
            if not result.need("jllama-build.txt" in inside, f"{classifier}: no jllama-build.txt"):
                continue
            build = build_info(jar.read(prefix + "jllama-build.txt").decode("utf-8"))
            builds[classifier] = build
            missing = [k for k in BUILD_KEYS if not build.get(k)]
            result.need(not missing, f"{classifier}: jllama-build.txt lacks {missing}")
            for key, expected in (("classifier", classifier), ("kind", kind), ("jllama.version", version)):
                result.need(build.get(key) == expected, f"{classifier}: jllama-build.txt says {key}={build.get(key)}, "
                                                        f"expected {expected}")
            name = manifest_attribute(jar.read("META-INF/MANIFEST.MF").decode("utf-8"), "Automatic-Module-Name")
            result.need(name == natives.module_name(classifier),
                        f"{classifier}: Automatic-Module-Name {name}, expected {natives.module_name(classifier)}")
            result.notes.append(f"{classifier}: {len(inside)} files in {directory}")
    result.notes.insert(0, f"{len(builds)} of {len(rows)} natives jars read")
    return result, builds


# --- provenance ---------------------------------------------------------------------------------

def git(root, *args):
    return subprocess.run(["git", "-C", root, *args], capture_output=True, check=True).stdout


def patch_set(root, revision="HEAD"):
    """[(name, sha256)] of llama/patches/*.patch|*.diff at revision, in name order -- the blobs, not
    the working tree, so a checkout's line-end conversion cannot change a hash."""
    names = git(root, "ls-tree", "--name-only", f"{revision}:llama/patches").decode().split()
    names = sorted(n for n in names if n.endswith((".patch", ".diff")))
    return [(n, hashlib.sha256(git(root, "show", f"{revision}:llama/patches/{n}")).hexdigest()) for n in names]


def patches_sha256(patches):
    """jllama-build-info.cmake's hash: SHA-256 over "<name> <sha256>\\n" per patch."""
    return hashlib.sha256("".join(f"{n} {h}\n" for n, h in patches).encode()).hexdigest()


def patch_ids(patches):
    return ",".join(re.sub(r"^([0-9]+)-.*", r"\1", n) for n, _ in patches)


def check_provenance(builds, version, commit, patches):
    result = Result("provenance")
    if not result.need(bool(builds), "no jllama-build.txt was read (see natives)"):
        return result
    for key in ("llama.cpp", "llama.cpp.commit"):
        values = {b.get(key) for b in builds.values()}
        result.need(len(values) == 1, f"{key} differs between the natives jars: {sorted(map(str, values))}")
    expected_hash, expected_ids = patches_sha256(patches), patch_ids(patches)
    for classifier, build in sorted(builds.items()):
        result.need(build.get("jllama.version") == version, f"{classifier}: jllama.version {build.get('jllama.version')}")
        result.need(build.get("jllama.commit") == commit,
                    f"{classifier}: jllama.commit {build.get('jllama.commit')}, the tag is {commit}")
        result.need(build.get("patches") == expected_ids, f"{classifier}: patches={build.get('patches')}, "
                                                          f"the tag has {expected_ids}")
        result.need(build.get("patches.sha256") == expected_hash,
                    f"{classifier}: patches.sha256 {build.get('patches.sha256')}, recomputed from the tag {expected_hash}")
    first = next(iter(builds.values()))
    result.notes.append(f"llama.cpp {first.get('llama.cpp')} ({first.get('llama.cpp.commit')}), {len(patches)} patches, "
                        f"patches.sha256 {expected_hash[:16]}… recomputed from the tag")
    toolkits = sorted({f"{c}: {b.get('toolkit')}" for c, b in builds.items() if b.get("kind") == "module"})
    if toolkits:
        result.notes.append("toolkits: " + "; ".join(toolkits))
    signed = sorted({b.get("signed") for b in builds.values()})
    result.notes.append(f"signed={','.join(map(str, signed))} (docs/post-release-check.md, check 16)")
    return result


# --- the source archive -------------------------------------------------------------------------

def added_lines(patch_text):
    """{path: [added lines]} of a unified diff, new files and modified files alike."""
    out, current = {}, None
    for line in patch_text.splitlines():
        if line.startswith("+++ "):
            target = line[4:].split("\t")[0]
            current = None if target == "/dev/null" else re.sub(r"^b/", "", target)
            if current:
                out.setdefault(current, [])
        elif current and line.startswith("+") and not line.startswith("+++"):
            out[current].append(line[1:])
    return out


def check_source(assets, version, builds, patches, patch_text):
    """patch_text(name) -> the patch at the tag, as text."""
    result = Result("source")
    archives = [n for n in os.listdir(assets) if re.fullmatch(rf"llama\.cpp-.+-jllama-{re.escape(version)}-src\.tar\.gz", n)]
    if not result.need(len(archives) == 1, f"expected one source archive, found {archives or 'none'}"):
        return result
    with tarfile.open(os.path.join(assets, archives[0])) as tar:
        members = tar.getmembers()
        top = archives[0][:-len(".tar.gz")]
        result.need(all(m.name == top or m.name.startswith(top + "/") for m in members),
                    f"not everything is below {top}/")
        mtimes = {m.mtime for m in members}
        result.need(len(mtimes) == 1, f"{len(mtimes)} different mtimes (a reproducible pack has one)")
        owners = {(m.uid, m.gid, m.uname, m.gname) for m in members}
        result.need(owners == {(0, 0, "", "")}, f"owners {sorted(owners)[:3]}, expected numeric 0/0 without names")
        files = {m.name[len(top) + 1:]: m for m in members if m.isfile()}
        if not result.need(".jllama-patches-applied" in files, "no .jllama-patches-applied stamp"):
            return result
        stamp = tar.extractfile(files[".jllama-patches-applied"]).read().decode().splitlines()
        head = next((line[5:] for line in stamp if line.startswith("head ")), None)
        commits = {b.get("llama.cpp.commit") for b in builds.values()}
        result.need(commits == {head}, f"the stamp's head {head}, the natives jars say {sorted(map(str, commits))}")
        stamped = [tuple(line.split(" ", 1)) for line in stamp if re.match(r"\S+\.(patch|diff) [0-9a-f]{64}$", line)]
        result.need(stamped == patches, "the stamp's patches differ from the tag's: "
                    f"only stamped {sorted(set(stamped) - set(patches))[:3]}, only at the tag {sorted(set(patches) - set(stamped))[:3]}")
        # A later patch may rewrite what an earlier one added (0006/0007 rewrite 0001's lines in
        # server.cpp), so a file is held to the LAST patch touching it; a patch whose every file is
        # touched again later needs one of its added lines in one of its files.
        added = [(name, added_lines(patch_text(name))) for name, _ in patches]
        last = {path: name for name, paths in added for path in paths}
        texts = {}

        def text(path):
            if path not in texts:
                member = files.get(path)
                texts[path] = tar.extractfile(member).read().decode("utf-8", "replace") if member else None
            return texts[path]

        for name, paths in added:
            wanted = {p: [line.strip() for line in lines if len(line.strip()) >= 12] for p, lines in paths.items()}
            wanted = {p: lines for p, lines in wanted.items() if lines}
            if not wanted:
                continue
            for path in wanted:
                result.need(text(path) is not None, f"{name}: {path} is not in the archive")
            own = [p for p in wanted if last[p] == name and text(p) is not None]
            for path in own:
                result.need(any(line in text(path) for line in wanted[path]), f"{name}: none of its added lines is in {path}")
            if not own:
                result.need(any(text(p) and any(line in text(p) for line in lines) for p, lines in wanted.items()),
                            f"{name}: none of its added lines survives in {sorted(wanted)}")
        result.notes.append(f"{archives[0]}: {len(members)} entries, head {head}, {len(stamped)} patches stamped")
    return result


# --- Android ------------------------------------------------------------------------------------

def zip_entries(data):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return {i.filename: i.CRC for i in archive.infolist() if not i.filename.endswith("/")}


def check_android(assets, version, root):
    result = Result("android")
    cpu_path = os.path.join(assets, f"llama-android-{version}.aar")
    opencl_path = os.path.join(assets, f"llama-android-opencl-{version}.aar")
    core_path = os.path.join(assets, f"llama-{version}.jar")
    for path in (cpu_path, opencl_path, core_path):
        if not result.need(os.path.isfile(path), f"{os.path.basename(path)} is not in the release"):
            return result
    elves = []
    with zipfile.ZipFile(cpu_path) as cpu, zipfile.ZipFile(opencl_path) as opencl:
        for name, aar in (("llama-android", cpu), ("llama-android-opencl", opencl)):
            manifest = aar.read("AndroidManifest.xml").decode("utf-8")
            result.need(f'android:minSdkVersion="{MIN_SDK}"' in manifest, f"{name}: minSdkVersion is not {MIN_SDK}")
            result.need("proguard.txt" in aar.namelist() and aar.read("proguard.txt").strip(),
                        f"{name}: no consumer rules (proguard.txt)")
            elves += [(f"{name}.aar!{n}", aar.read(n)) for n in _files(aar) if n.endswith(".so")]
        with open(core_path, "rb") as stream:
            core = {n: c for n, c in zip_entries(stream.read()).items() if not n.endswith("module-info.class")}
        classes = zip_entries(cpu.read("classes.jar"))
        core_classes = {n: c for n, c in core.items() if n.endswith(".class")}
        aar_classes = {n: c for n, c in classes.items() if n.endswith(".class")}
        result.need(aar_classes == core_classes, "llama-android: classes.jar is not the classes jar minus module-info "
                    f"({len(aar_classes)} classes, the classes jar has {len(core_classes)}; "
                    f"differing {sorted(set(aar_classes.items()) ^ set(core_classes.items()))[:2]})")
        rows = read_rows(root)
        for abi, classifier in ANDROID_ABIS.items():
            row = next(r for r in rows if r["classifier"] == classifier)
            with zipfile.ZipFile(os.path.join(assets, f"llama-{version}-{classifier}.jar")) as jar:
                prefix = f"net/ladenthin/llama/{row['directory']}/"
                expected = {n[len(prefix):]: jar.getinfo(n).CRC for n in _files(jar)
                            if n.startswith(prefix) and n.endswith(".so")}
                elves += [(f"{classifier}!{n}", jar.read(prefix + n)) for n in expected]
            actual = {n[len(f"jni/{abi}/"):]: cpu.getinfo(n).CRC for n in _files(cpu) if n.startswith(f"jni/{abi}/")}
            result.need(actual == expected, f"llama-android jni/{abi}/ differs from {classifier}: "
                        f"{sorted(set(actual.items()) ^ set(expected.items()))[:3]}")
        jni = sorted(n for n in _files(opencl) if n.startswith("jni/"))
        result.need(jni == ["jni/arm64-v8a/libggml-opencl.so"], f"llama-android-opencl: jni/ holds {jni}")
        result.need(not any(n.endswith(".class") for n in zip_entries(opencl.read("classes.jar"))),
                    "llama-android-opencl: classes.jar is not empty (duplicating the API breaks D8)")
    for name, data in elves:
        bad = [a for a in nativedeps.elf_load_alignments(data) if a % nativedeps.ANDROID_PAGE_ALIGNMENT]
        result.need(not bad, f"{name}: LOAD alignment {bad} (Google Play needs 16 KB)")
    expected = aarpoms.core_compile_dependencies(os.path.join(root, "llama", "pom.xml"), os.path.join(root, "pom.xml"))
    for artifact in ("llama-android", "llama-android-opencl"):
        pom = os.path.join(assets, f"{artifact}-{version}.pom")
        if not result.need(os.path.isfile(pom), f"{artifact}-{version}.pom is not in the release"):
            continue
        published = aarpoms.published_dependencies(pom)
        result.problems += [f"{artifact} pom: {p}" for p in aarpoms.violations(expected, published)]
        if artifact == "llama-android-opencl":
            cpu_dep = [d for d in published if (d[0], d[1]) == aarpoms.CPU_AAR]
            result.need([d[2] for d in cpu_dep] == [version],
                        f"llama-android-opencl pom: the CPU AAR dependency is {cpu_dep}, expected version {version}")
    result.notes.append(f"{len(elves)} Android ELF files 16 KB aligned; {len(core_classes)} classes in the CPU AAR")
    return result


# --- the cache-free tag build -------------------------------------------------------------------

CACHE_ON = re.compile(r"USE_CACHE: true|build\.sh: sccache ON|build\.bat: sccache probe OK")
CACHE_OFF = re.compile(r"USE_CACHE: false|build\.sh: sccache OFF")


def check_cache(logs):
    """logs: {job name: log text} of the publishing run."""
    result = Result("cache")
    off = 0
    for name, text in sorted(logs.items()):
        hits = sorted({m.group(0) for m in CACHE_ON.finditer(text)})
        result.need(not hits, f"{name}: {', '.join(hits)}")
        off += bool(CACHE_OFF.search(text))
    result.need(off > 0, "no job log shows the cache off -- the evidence is missing, not the cache")
    result.notes.append(f"{len(logs)} job logs read, {off} show the cache off, none on")
    return result


def run_logs(repository, run_id):
    import json
    jobs, page = [], 1
    while True:
        out = subprocess.run(["gh", "api", f"repos/{repository}/actions/runs/{run_id}/jobs?filter=latest&per_page=50&page={page}"],
                             capture_output=True, text=True, check=True).stdout
        batch = json.loads(out)["jobs"]
        jobs += batch
        if len(batch) < 50:
            break
        page += 1
    logs = {}
    for job in jobs:
        if job["conclusion"] == "skipped":
            continue
        out = subprocess.run(["gh", "api", f"repos/{repository}/actions/jobs/{job['id']}/logs"],
                             capture_output=True, check=True).stdout
        logs[job["name"]] = out.decode("utf-8", "replace")
    return logs


# --- the command --------------------------------------------------------------------------------

def find_run(repository, tag):
    import json
    out = subprocess.run(["gh", "api", f"repos/{repository}/actions/workflows/publish.yml/runs?branch={tag}&per_page=50"],
                         capture_output=True, text=True, check=True).stdout
    good = [r["id"] for r in json.loads(out)["workflow_runs"] if r["conclusion"] == "success"]
    if len(good) != 1:
        raise ValueError(f"expected one successful Publish run on {tag}, found {good}")
    return good[0]


def main(argv):
    parser = argparse.ArgumentParser(prog="post-release-check.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=".", help="a checkout of the tag (default: the current directory)")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--assets", required=True, help="the GitHub release files, downloaded")
    parser.add_argument("--run", help="the publishing run (default: the one successful Publish run on the tag)")
    parser.add_argument("--checks", default=",".join(CHECKS))
    args = parser.parse_args(argv[1:])
    checks = [c for c in args.checks.split(",") if c]
    if set(checks) - set(CHECKS):
        parser.error(f"unknown check(s) {sorted(set(checks) - set(CHECKS))}; known: {', '.join(CHECKS)}")
    root = os.path.abspath(args.root)
    rows = read_rows(root)
    version = args.tag[1:]
    commit = git(root, "rev-parse", "HEAD").decode().strip()
    patches = patch_set(root)
    results = []
    natives_result, builds = check_natives(args.assets, rows, version)
    if "natives" in checks:
        results.append(natives_result)
    if "provenance" in checks:
        results.append(check_provenance(builds, version, commit, patches))
    if "source" in checks:
        results.append(check_source(args.assets, version, builds, patches,
                                    lambda n: git(root, "show", f"HEAD:llama/patches/{n}").decode("utf-8", "replace")))
    if "android" in checks:
        results.append(check_android(args.assets, version, root))
    if "cache" in checks:
        repository = os.environ.get("GITHUB_REPOSITORY", "bernardladenthin/java-llama.cpp")
        results.append(check_cache(run_logs(repository, args.run or find_run(repository, args.tag))))
    failed = 0
    for result in results:
        print(f"== {result.name}: {'FAIL' if result.problems else 'ok'}")
        for note in result.notes:
            print(f"   {note}")
        for problem in result.problems:
            print(f"::error::{result.name}: {problem}")
        failed += bool(result.problems)
    print(f"post-release checks of {args.tag}: {len(results)} checks, {failed} failed")
    return 1 if failed else 0
