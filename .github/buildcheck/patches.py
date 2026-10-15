# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The llama/patches/*.patch files, checked as TEXT before any job applies them: every hunk header
declares exactly the lines its body carries.

Why a check of the text, when the applier is fail-loud. `git apply` (and `git apply --check`) reads
a hunk by its header counts and then looks for the next header. Lines left over after the counted
body are not an error to it: they are skipped as the start of the next (non-existent) header. So a
new-file hunk that says `@@ -0,0 +1,25 @@` over 27 `+` lines applies "cleanly" and writes the first
25 lines -- the file is truncated, silently. That is what #489 shipped (two comment lines added to
`prefetch.h` inside patch 0017 without touching the count): `prefetch.h` ended in the middle of its
one function, every translation unit including it failed, and run 37922404867 went red in 24 jobs
before #492 corrected the count. Checking that a patch applies is not checking that it applies
correctly; this module checks the half `git apply --check` cannot, in the `code-style` job, in the
first minutes of every run and with no llama.cpp source needed.

Two shapes are reported, for every hunk of every patch:
  * the body ends before the header's counts are satisfied -- a header that claims too much, or a
    patch cut short; `git apply` rejects this one as corrupt, so it is caught here earlier, not only;
  * lines that look like body (`+`, `-` or context) follow a satisfied hunk before the next header --
    the #489 shape, which `git apply` accepts and truncates.
The patch header text before the first `diff --git` is free prose and is not inspected.
"""

import glob
import os
import re

PATCH_DIR = os.path.join("llama", "patches")
DOC = os.path.join("docs", "patches.md")
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
# What may follow a complete hunk: the next hunk, the next file's header, or the format-patch
# signature trailer (`-- ` alone or followed by the git version).
HEADER_PREFIXES = ("diff --git ", "index ", "--- ", "+++ ", "new file mode ", "deleted file mode ", "old mode ",
                   "new mode ", "similarity index ", "dissimilarity index ", "rename from ", "rename to ",
                   "copy from ", "copy to ", "Binary files ")
TRAILER = re.compile(r"^-- (\d|$)")


def _is_header(line):
    return bool(HUNK.match(line)) or line.startswith(HEADER_PREFIXES) or bool(TRAILER.match(line))


def audit(text, name="<patch>"):
    """(problems, hunks) of one patch's text: the problems as messages naming the patch, the line and
    the file the hunk belongs to, and the number of hunks seen."""
    problems = []
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    hunks = 0
    target = "?"
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("+++ "):
            target = line[4:].strip()
            target = target[2:] if target.startswith("b/") else target
        match = HUNK.match(line)
        if not match:
            i += 1
            continue
        hunks += 1
        header_line = i + 1
        old = int(match.group(2)) if match.group(2) is not None else 1
        new = int(match.group(4)) if match.group(4) is not None else 1
        o = n = 0
        i += 1
        while i < len(lines) and (o < old or n < new):
            body = lines[i]
            if body.startswith("\\"):  # "\ No newline at end of file" belongs to the line before it
                i += 1
                continue
            if body.startswith("+"):
                n += 1
            elif body.startswith("-"):
                o += 1
            elif body.startswith(" ") or body == "":
                o += 1
                n += 1
            else:
                break
            i += 1
        if o != old or n != new:
            problems.append(f"{name}:{header_line}: the hunk for {target} declares -{old},+{new} lines but its body "
                            f"ends after -{o},+{n} -- a header claiming too much or a patch cut short; `git apply` "
                            f"rejects it as corrupt")
            continue
        extra = 0
        while i < len(lines) and not _is_header(lines[i]):
            body = lines[i]
            if body.startswith(("+", "-", " ")):
                extra += 1
            elif body != "" and not body.startswith("\\"):
                break
            i += 1
        if extra:
            problems.append(f"{name}:{header_line}: {extra} line(s) after the hunk for {target} are not covered by "
                            f"its header (-{old},+{new}) -- `git apply` drops them silently and writes a truncated "
                            f"file; recount the header")
    return problems, hunks


def check(root, patch_dir=PATCH_DIR):
    """(problems, patches, hunks) over every *.patch of the directory, in name order."""
    problems = []
    paths = sorted(glob.glob(os.path.join(root, patch_dir, "*.patch")))
    hunks = 0
    for path in paths:
        with open(path, "rb") as f:
            text = f.read().decode("utf-8", errors="surrogateescape")
        found, count = audit(text, os.path.relpath(path, root).replace(os.sep, "/"))
        problems += found
        hunks += count
    return problems, len(paths), hunks


# --- the rendered page -----------------------------------------------------------------------------
#
# docs/patches.md is generated from the patch files, never written by hand: a consumer who asks
# "what does a java-llama.cpp binary carry that upstream b<nnnn> does not" gets one page that cannot
# drift from the patches, because `code-style` fails when the committed page differs from what
# `check-patches.py --render` writes (TODO.md, delivery roadmap, item 1). jllama-build.txt in every
# natives directory names the same patch ids, so the page and the artefact describe the same set.

DOC_HEAD = """<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT
-->

# Patches carried on top of llama.cpp

<!-- GENERATED by `python3 .github/check-patches.py --render` from llama/patches/*.patch -- do not edit;
     the code-style job fails when this page is stale. -->

Every native library this project ships is built from the pinned llama.cpp source with the patches
below applied, in this order (`llama/cmake/apply-llama-patches.cmake`). They are the difference
between a java-llama.cpp binary and upstream's: bug fixes upstream has not merged yet, and the
embedding glue a JVM needs that upstream will not carry. Each is submitted upstream where it is
upstream-shaped, and dropped here the moment upstream carries the fix (the drop-checks are in
`CLAUDE.md`, "Local llama.cpp source patches"). Every natives jar names the ids it was built with
in its `jllama-build.txt` (`patches=`) and a hash over the set (`patches.sha256=`), so a consumer can
match a binary to this page.
"""

FENCE = "`" * 3


def describe(text, name):
    """{name, id, title, prose, files, hunks} of one patch's text: the prose before the first
    `diff --git` (its first non-empty line is the title), the files it touches, the hunk count."""
    prose_lines = []
    files = []
    for line in text.split("\n"):
        if line.startswith("diff --git "):
            break
        prose_lines.append(line)
    for line in text.split("\n"):
        if line.startswith("+++ "):
            target = line[4:].strip()
            if target.startswith("b/"):
                target = target[2:]
            if target != "/dev/null" and target not in files:
                files.append(target)
    prose = "\n".join(prose_lines).strip()
    title = next((l.strip() for l in prose_lines if l.strip()), name)
    patch_id = re.match(r"^(\d+)", name)
    _, hunks = audit(text, name)
    return {"name": name, "id": patch_id.group(1) if patch_id else name, "title": title, "prose": prose,
            "files": files, "hunks": hunks}


def render(root, patch_dir=PATCH_DIR):
    """The Markdown of docs/patches.md for the patches of the repository, in name order."""
    paths = sorted(glob.glob(os.path.join(root, patch_dir, "*.patch")))
    entries = []
    for path in paths:
        with open(path, "rb") as f:
            text = f.read().decode("utf-8", errors="replace")
        entries.append(describe(text, os.path.basename(path)))
    out = [DOC_HEAD]
    out.append(f"**{len(entries)} patches.** Ids as `jllama-build.txt` lists them under `patches=`.\n")
    out.append("| Id | Patch | Title | Files | Hunks |")
    out.append("|---|---|---|---:|---:|")
    for e in entries:
        anchor = e["name"].lower().replace(".", "")
        title = e["title"].replace("|", "\\|")
        out.append(f"| {e['id']} | [`{e['name']}`](#{anchor}) | {title} | {len(e['files'])} | {e['hunks']} |")
    out.append("")
    for e in entries:
        out.append(f"## {e['name']}")
        out.append("")
        out.append(f"**Files ({len(e['files'])}):** " + ", ".join(f"`{f}`" for f in e["files"]))
        out.append("")
        out.append("The patch's own header, verbatim:")
        out.append("")
        out.append(FENCE + "text")
        out.append(e["prose"].replace(FENCE, "'''"))
        out.append(FENCE)
        out.append("")
    return "\n".join(out)


def doc_is_current(root, patch_dir=PATCH_DIR, doc=DOC):
    """(current, path): whether the committed page equals what render() produces."""
    path = os.path.join(root, doc)
    try:
        with open(path, "rb") as f:
            committed = f.read().decode("utf-8", errors="replace")
    except OSError:
        return False, path
    return committed == render(root, patch_dir), path
