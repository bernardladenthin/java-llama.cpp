# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The license texts of the npm packages compiled into the embedded llama.cpp WebUI.

build-webui bundles llama.cpp's tools/ui with npm and embeds the result in every library natives
jar (webui-generated/ui.cpp), and the bundle strips every legal comment. So the texts travel in
llama/legal/natives/webui-npm/, generated from the pinned tools/ui:

- `closure()` -- a static upper bound of the packages the bundle can contain: the bare imports
  of tools/ui/src (tests, stories and type-only imports left out), closed over the runtime
  dependencies package-lock.json records. Upper bound on purpose: shipping a text for a package
  that tree-shaking dropped is harmless, missing one is not.
- `generate()` -- each package's own license/notice files, byte for byte from the tarball
  package-lock.json names (nothing is executed), concatenated into THIRD-PARTY-LICENSES.txt; a
  package without a license file gets its declared SPDX id and the author package.json names, and
  the file ends with the full text of every such id (from the repository's LICENSES/), so an MIT
  package without a LICENSE file still has its license text in the jar.
- `check()` -- build-webui compares the lockfile it builds with the one the texts came from
  (LOCKFILE.sha256), recomputes the closure from the sources it builds (a new import of a package
  the lockfile already had changes no hash), and holds packages.csv and the text file to each other:
  a section for every package and file it names, the full text of every declared-only license.
"""

import csv
import hashlib
import io
import json
import os
import re
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# A file at the tarball root (npm packs `package/`, DefinitelyTyped the package name).
LEGAL = re.compile(r"^[^/]+/(licen[cs]e|copying|notice|unlicense|authors)([.-][^/]*)?$", re.I)
QUOTE, NOT_QUOTE = "['\"]", "[^'\"]"
SPECIFIER = re.compile(
    r"(?:import|export)\s" + NOT_QUOTE + r"*?from\s*" + QUOTE + "(" + NOT_QUOTE + "+)" + QUOTE
    + r"|import\s*\(\s*" + QUOTE + "(" + NOT_QUOTE + "+)" + QUOTE + r"\s*\)"
    + r"|import\s+" + QUOTE + "(" + NOT_QUOTE + "+)" + QUOTE
    + r"|@import\s+" + QUOTE + "(" + NOT_QUOTE + "+)" + QUOTE)
TYPE_ONLY = re.compile(r"(?m)^\s*(?:import|export)\s+type\s[^;]*?from\s*" + QUOTE + NOT_QUOTE + "+" + QUOTE)
SKIPPED = re.compile(r"\.(spec|test|stories)\.|/tests?/|\.stories\.svelte$")
# Not imported by name, but part of every bundle the build produces.
IMPLICIT = ("svelte", "@sveltejs/kit", "workbox-window")
DECLARED = "(declared in package.json)"
# The SPDX texts for a package that ships no license file: the repository's REUSE folder.
SPDX_TEXTS = Path(__file__).resolve().parents[2] / "LICENSES"
SPDX_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]*")


def lockfile_digest(ui):
    return hashlib.sha256((Path(ui) / "package-lock.json").read_bytes()).hexdigest()


def imports(ui):
    """Bare package names imported by tools/ui/src."""
    roots = set(IMPLICIT)
    for folder, _, files in os.walk(Path(ui) / "src"):
        for name in files:
            path = (Path(folder) / name).as_posix()
            if (not re.search(r"\.(ts|js|svelte|css|mjs)$", name) or name.endswith(".d.ts")
                    or SKIPPED.search(path)):
                continue
            source = TYPE_ONLY.sub("", Path(path).read_text(encoding="utf-8", errors="replace"))
            for match in SPECIFIER.finditer(source):
                spec = next(group for group in match.groups() if group)
                if spec.startswith((".", "/", "$", "virtual:", "node:")):
                    continue
                parts = spec.split("/")
                roots.add("/".join(parts[:2]) if spec.startswith("@") else parts[0])
    return roots


def closure(lock, roots):
    """The lockfile entries reachable from `roots` through dependencies/optionalDependencies."""
    packages = lock["packages"]

    def resolve(name, start):
        base = start
        while True:
            candidate = (base + "/" if base else "") + "node_modules/" + name
            if candidate in packages:
                return candidate
            if not base:
                return None
            cut = base.rfind("/node_modules/")
            base = base[:cut] if cut >= 0 else ""

    seen, stack = {}, [(key, None) for key in (resolve(root, "") for root in sorted(roots)) if key]
    while stack:
        key, via = stack.pop()
        if key in seen:
            continue
        seen[key] = via
        entry = packages[key]
        for field in ("dependencies", "optionalDependencies"):
            for dependency in sorted(entry.get(field) or {}):
                found = resolve(dependency, key)
                if found:
                    stack.append((found, key))
    result, listed = [], set()
    for key in sorted(seen):
        entry = packages[key]
        name, version = key.split("node_modules/")[-1], entry.get("version", "")
        # Type declarations are erased by the build: nothing of an @types package reaches the bundle.
        if name.startswith("@types/") or (name, version) in listed:
            continue
        listed.add((name, version))
        result.append({"name": name, "version": version,
                       "license": entry.get("license") or "", "resolved": entry.get("resolved") or ""})
    return result


def _texts(package, fetch):
    """The package's own legal files from its registry tarball, or its declared license + author."""
    url = package["resolved"]
    if not url.startswith("https://registry.npmjs.org/"):
        raise ValueError(f"{package['name']}@{package['version']}: no registry tarball ({url})")
    texts, author = [], ""
    with tarfile.open(fileobj=io.BytesIO(fetch(url)), mode="r:gz") as archive:
        for member in sorted(archive.getmembers(), key=lambda m: m.name):
            if member.isfile() and LEGAL.match(member.name):
                texts.append((member.name.split("/", 1)[1], archive.extractfile(member).read()))
            elif member.isfile() and member.name.count("/") == 1 and member.name.endswith("/package.json"):
                meta = json.loads(archive.extractfile(member).read())
                author = meta.get("author") or ""
                if isinstance(author, dict):
                    author = " ".join(str(author[k]) for k in ("name", "email") if author.get(k))
    if not texts:
        declared = (f"License (package.json): {package['license'] or 'not declared'}\n"
                    f"Author (package.json): {author or 'not declared'}\n"
                    "(The package ships no license file; this is the license and author its "
                    "package.json declares; its full text is at the end of this file.)\n")
        texts.append((DECLARED, declared.encode("utf-8")))
    return texts


def declared_ids(expression):
    """The SPDX license ids of a package.json license expression (operators left out)."""
    return [token for token in SPDX_ID.findall(expression or "") if token not in ("AND", "OR", "WITH")]


def _section(title):
    return f"\n{'=' * 78}\n{title}\n{'=' * 78}\n"


def _spdx_title(spdx_id):
    return f"License text {spdx_id} (for the packages above marked {DECLARED})"


def generate(ui, out, fetch=None, workers=12):
    """Write THIRD-PARTY-LICENSES.txt, packages.csv and LOCKFILE.sha256 into `out`."""
    fetch = fetch or (lambda url: urllib.request.urlopen(url, timeout=120).read())
    ui, out = Path(ui), Path(out)
    lock = json.loads((ui / "package-lock.json").read_text(encoding="utf-8"))
    packages = closure(lock, imports(ui))
    with ThreadPoolExecutor(workers) as pool:
        texts = list(pool.map(lambda package: _texts(package, fetch), packages))
    undeclared = [f"{package['name']}@{package['version']}" for package, files in zip(packages, texts)
                  if files[0][0] == DECLARED and not declared_ids(package["license"])]
    if undeclared:
        raise ValueError(f"ships no license file and declares no license: {', '.join(undeclared)}")
    fallback = sorted({spdx_id for package, files in zip(packages, texts) if files[0][0] == DECLARED
                       for spdx_id in declared_ids(package["license"])})
    missing = [spdx_id for spdx_id in fallback if not (SPDX_TEXTS / f"{spdx_id}.txt").is_file()]
    if missing:
        raise ValueError(f"no text for {', '.join(missing)} in {SPDX_TEXTS}: add the SPDX text there")
    out.mkdir(parents=True, exist_ok=True)
    body = io.StringIO()
    body.write("Licenses of the npm packages the embedded llama.cpp WebUI can contain (an upper bound,\n"
               "see .github/buildcheck/webuilicenses.py). Generated; do not edit.\n")
    for package, files in zip(packages, texts):
        for name, data in files:
            body.write(_section(f"{package['name']}@{package['version']} ({package['license']}): {name}"))
            body.write(data.decode("utf-8", errors="replace").replace("\r\n", "\n").rstrip("\n") + "\n")
    for spdx_id in fallback:
        body.write(_section(_spdx_title(spdx_id)))
        text = (SPDX_TEXTS / f"{spdx_id}.txt").read_text(encoding="utf-8")
        body.write(text.replace("\r\n", "\n").rstrip("\n") + "\n")
    (out / "THIRD-PARTY-LICENSES.txt").write_bytes(body.getvalue().encode("utf-8"))
    with (out / "packages.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["package", "version", "license", "files", "tarball"])
        for package, files in zip(packages, texts):
            writer.writerow([package["name"], package["version"], package["license"],
                             ";".join(name for name, _ in files), package["resolved"]])
    (out / "LOCKFILE.sha256").write_bytes(f"{lockfile_digest(ui)}  tools/ui/package-lock.json\n".encode("utf-8"))
    return packages


def _listed(out):
    with (Path(out) / "packages.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def check_texts(out):
    """packages.csv and THIRD-PARTY-LICENSES.txt describe each other: one section per listed file."""
    text = (Path(out) / "THIRD-PARTY-LICENSES.txt").read_text(encoding="utf-8")
    problems, needed = [], set()
    for row in _listed(out):
        package = f"{row['package']}@{row['version']}"
        files = [name for name in row["files"].split(";") if name]
        if not files:
            problems.append(f"{package}: packages.csv names no license file")
        for name in files:
            if _section(f"{package} ({row['license']}): {name}") not in text:
                problems.append(f"{package}: no section for {name} in THIRD-PARTY-LICENSES.txt")
            if name == DECLARED:
                ids = declared_ids(row["license"])
                needed.update(ids)
                if not ids:
                    problems.append(f"{package}: ships no license file and declares no license")
    for spdx_id in sorted(needed):
        if _section(_spdx_title(spdx_id)) not in text:
            problems.append(f"no {spdx_id} license text in THIRD-PARTY-LICENSES.txt for the packages "
                            "that only declare it")
    return problems


def check(ui, out):
    """[] when the texts in `out` cover this tools/ui: its lockfile, its imports, every listed file."""
    regenerate = ("run `python3 .github/webui-licenses.py generate <llama.cpp>/tools/ui "
                  "llama/legal/natives/webui-npm` and commit the result")
    recorded = (Path(out) / "LOCKFILE.sha256").read_text(encoding="utf-8").split()[0]
    actual = lockfile_digest(ui)
    problems = []
    if recorded != actual:
        problems.append(f"tools/ui/package-lock.json is {actual}, the WebUI license texts were generated "
                        f"from {recorded}: {regenerate}")
    # A new import of a package the lockfile already had changes no hash.
    lock = json.loads((Path(ui) / "package-lock.json").read_text(encoding="utf-8"))
    listed = {(row["package"], row["version"]) for row in _listed(out)}
    absent = [f"{package['name']}@{package['version']}" for package in closure(lock, imports(ui))
              if (package["name"], package["version"]) not in listed]
    if absent:
        problems.append(f"{len(absent)} package(s) the WebUI sources can reach are not in packages.csv "
                        f"({', '.join(absent[:10])}{', ...' if len(absent) > 10 else ''}): {regenerate}")
    return problems + check_texts(out)
