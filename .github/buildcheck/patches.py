# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Fail when a patch in llama/patches/ declares a hunk size that does not match its body.

A unified-diff hunk header `@@ -a,b +c,d @@` promises exactly `b` old-side and `d` new-side lines.
When the body holds MORE than that, `git apply` writes only the declared number and silently
truncates -- and `git apply --check` does NOT notice, because it validates that the context applies,
not that the counts add up. That is not hypothetical: editing the comment inside a new file added by
`patches/0017` left its header at `+1,25` for 27 lines, so the file was written without its closing
brace, every translation unit including it continued inside an unclosed function, and `quants.c`
failed with 19 x "function definition is not allowed here" on every platform. 23 jobs of one
Publish run went red for it, after `--check` had reported the patch as fine.

The applier (`llama/cmake/apply-llama-patches.cmake`) cannot catch this either: a truncated file is
a successfully applied patch as far as git is concerned, and `.github/verify-patches-applied.sh`
only asserts that every patch is in the stamp and that the tree is dirty -- both true here.

So this is a STATIC check of the patch files, cheap enough to run in the first minutes of a run
(the `code-style` job) and before a push.

**The algorithm is the part that matters, and two obvious forms of it are wrong.** Counting body
lines until the next delimiter over-counts trailing context *and* accepts a header that
under-counts its body -- exactly the bug above. Only checking that the body is not SHORT misses it
as well. The correct form is: consume exactly as many lines as the header declares, then require
that a delimiter follows. A body that ends early is short, a body that continues is under-counted,
and both are reported with what `git apply` would do about it.

Usage (the CLI is .github/check-patches.py): check-patches.py [<patch-dir>]
"""

import glob
import io
import os
import re
import sys

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
# "\ No newline at end of file" belongs to the line before it and counts for neither side.
NO_NEWLINE = chr(92)
DEFAULT_DIR = os.path.join("llama", "patches")


def _delimiter(line):
    """Whether `line` ends a hunk body: the next hunk, the next file, or a git signature."""
    return line.startswith("@@") or line.startswith("diff --git") or line == "-- "


def problems(path):
    """Every hunk of `path` whose declared size disagrees with its body, as messages."""
    lines = io.open(path, encoding="utf-8", newline="").read().splitlines()
    name = os.path.basename(path)
    found = []
    for i, line in enumerate(lines):
        m = HUNK.match(line)
        if not m:
            continue
        want_old = int(m.group(2)) if m.group(2) is not None else 1
        want_new = int(m.group(4)) if m.group(4) is not None else 1
        old = new = 0
        j = i + 1
        while j < len(lines) and (old < want_old or new < want_new):
            body = lines[j]
            if _delimiter(body):
                break
            if body.startswith(NO_NEWLINE):
                pass
            elif body.startswith("+"):
                new += 1
            elif body.startswith("-"):
                old += 1
            elif body.startswith(" ") or body == "":
                old += 1
                new += 1
            else:
                break  # prose below the last hunk of a patch that carries a description
            j += 1
        if (old, new) != (want_old, want_new):
            found.append(f"{name}:{i + 1}: hunk declares -{want_old} +{want_new} but its body holds "
                         f"{old} old / {new} new lines -- it is SHORT, so `git apply` fails or "
                         f"writes a partial file")
            continue
        if j < len(lines) and not _delimiter(lines[j]) and lines[j][:1] in ("+", "-", " "):
            found.append(f"{name}:{i + 1}: hunk declares -{want_old} +{want_new} but its body "
                         f"continues at line {j + 1} ({lines[j][:48]!r}) -- the header UNDER-counts, "
                         f"so `git apply` writes a TRUNCATED file and `--check` does not notice")
    return found


def main(argv):
    directory = argv[1] if len(argv) > 1 else DEFAULT_DIR
    paths = sorted(glob.glob(os.path.join(directory, "*.patch")) +
                   glob.glob(os.path.join(directory, "*.diff")))
    if not paths:
        print(f"::error::no patches found in {directory} -- is the path right?", file=sys.stderr)
        return 2
    found = []
    for p in paths:
        found += problems(p)
    for f in found:
        print(f"::error::{f}", file=sys.stderr)
    print(f"{len(paths)} patches checked, {len(found)} hunk size mismatch(es)")
    return 1 if found else 0
