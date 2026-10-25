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
  package without a license file gets its declared SPDX id and the author package.json names.
- `check()` -- build-webui compares the lockfile it builds with the one the texts came from
  (LOCKFILE.sha256), so a llama.cpp bump that changes the WebUI's packages cannot ship the old list.
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
                    "package.json declares. The SPDX license text applies.)\n")
        texts.append(("(declared in package.json)", declared.encode("utf-8")))
    return texts


def generate(ui, out, fetch=None, workers=12):
    """Write THIRD-PARTY-LICENSES.txt, packages.csv and LOCKFILE.sha256 into `out`."""
    fetch = fetch or (lambda url: urllib.request.urlopen(url, timeout=120).read())
    ui, out = Path(ui), Path(out)
    lock = json.loads((ui / "package-lock.json").read_text(encoding="utf-8"))
    packages = closure(lock, imports(ui))
    with ThreadPoolExecutor(workers) as pool:
        texts = list(pool.map(lambda package: _texts(package, fetch), packages))
    out.mkdir(parents=True, exist_ok=True)
    body = io.StringIO()
    body.write("Licenses of the npm packages the embedded llama.cpp WebUI can contain (an upper bound,\n"
               "see .github/buildcheck/webuilicenses.py). Generated; do not edit.\n")
    for package, files in zip(packages, texts):
        for name, data in files:
            body.write(f"\n{'=' * 78}\n{package['name']}@{package['version']} ({package['license']}): {name}\n"
                       f"{'=' * 78}\n")
            body.write(data.decode("utf-8", errors="replace").replace("\r\n", "\n").rstrip("\n") + "\n")
    (out / "THIRD-PARTY-LICENSES.txt").write_bytes(body.getvalue().encode("utf-8"))
    with (out / "packages.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["package", "version", "license", "files", "tarball"])
        for package, files in zip(packages, texts):
            writer.writerow([package["name"], package["version"], package["license"],
                             ";".join(name for name, _ in files), package["resolved"]])
    (out / "LOCKFILE.sha256").write_text(lockfile_digest(ui) + "  tools/ui/package-lock.json\n", encoding="utf-8")
    return packages


def check(ui, out):
    """[] when the texts in `out` were generated from this tools/ui/package-lock.json."""
    recorded = (Path(out) / "LOCKFILE.sha256").read_text(encoding="utf-8").split()[0]
    actual = lockfile_digest(ui)
    if recorded == actual:
        return []
    return [f"tools/ui/package-lock.json is {actual}, the WebUI license texts were generated from "
            f"{recorded}: run `python3 .github/webui-licenses.py generate <llama.cpp>/tools/ui "
            f"llama/legal/natives/webui-npm` and commit the result"]
