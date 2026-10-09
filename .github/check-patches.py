#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Fail when a hunk of llama/patches/*.patch does not declare exactly the lines it carries.

`git apply --check` does not catch this: lines past a hunk's declared count are skipped, which
truncates a new file silently (see buildcheck/patches.py for the incident). Runs in the
`code-style` job; needs no llama.cpp source.

Usage:
  check-patches.py        check, exit 1 on any problem
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from buildcheck import patches  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv):
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
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
