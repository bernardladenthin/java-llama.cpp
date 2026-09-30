#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Fail when anything that names the natives jars disagrees with .github/natives.csv (and
when a model name of publish.yml is not in .github/models.csv, see buildcheck/models.py).

See buildcheck/natives.py for what is checked.

Usage:
  check-natives.py                  check, exit 1 on any disagreement
  check-natives.py pom              print the natives jar executions for llama/pom.xml
  check-natives.py fatjar-targets   print the all-backends fat-jar targets, one per line
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from buildcheck import models, natives  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv):
    rows = natives.rows(natives.read(ROOT, ".github/natives.csv"))
    if argv[1:] == ["pom"]:
        print("\n".join(natives.pom_execution(r) for r in rows))
        return 0
    if argv[1:] == ["fatjar-targets"]:
        print("\n".join(natives.fatjar_targets(rows)))
        return 0
    if argv[1:]:
        print(__doc__, file=sys.stderr)
        return 2
    failures = natives.check(ROOT) + models.check(natives.read(ROOT, ".github/models.csv"),
                                                  natives.read(ROOT, ".github/workflows/publish.yml"))
    for f in failures:
        print(f"::error::{f}", file=sys.stderr)
    print(f"{len(rows)} natives jars, {len(natives.fatjar_targets(rows))} all-backends fat jars, "
          f"{len(failures)} disagreements")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
