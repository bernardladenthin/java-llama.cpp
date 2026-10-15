#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Fail when a hunk of llama/patches/*.patch does not declare exactly the lines it carries.

`git apply --check` does not catch this: lines past a hunk's declared count are skipped, which
truncates a new file silently (see buildcheck/patches.py for the incident). Runs in the
`code-style` job; needs no llama.cpp source.

Also holds docs/patches.md to the patches: the page is generated from their headers, and the check
fails when the committed page is not what `--render` writes.

Usage:
  check-patches.py           check the patches and the page, exit 1 on any problem
  check-patches.py --render  (re)write docs/patches.md from the patches
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from buildcheck import patches  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv):
    if argv[1:] == ["--render"]:
        path = os.path.join(ROOT, patches.DOC)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(patches.render(ROOT))
        print(f"wrote {path}")
        return 0
    if argv[1:]:
        print(__doc__, file=sys.stderr)
        return 2
    problems, count, hunks = patches.check(ROOT)
    for p in problems:
        print(f"::error::{p}", file=sys.stderr)
    print(f"{count} patches, {hunks} hunks, {len(problems)} problems")
    if count == 0:
        print("::error::no patches found -- an empty input is a failure, never a pass", file=sys.stderr)
        return 2
    current, path = patches.doc_is_current(ROOT)
    if not current:
        print(f"::error::{os.path.relpath(path, ROOT)} is not what the patches say -- run "
              f"`python3 .github/check-patches.py --render` and commit the page", file=sys.stderr)
        return 1
    print(f"{os.path.relpath(path, ROOT)} is current")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
