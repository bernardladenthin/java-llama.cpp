#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Fail when anything that names the natives jars disagrees with .github/natives.csv (and
when a model name of publish.yml is not in .github/models.csv, see buildcheck/models.py).

See buildcheck/natives.py for what is checked.

Usage:
  check-natives.py                  check, exit 1 on any disagreement
  check-natives.py pom              print the natives jar executions for llama/pom.xml
  check-natives.py smoke-targets    print the smoke targets (<os>-<arch>), one per line
  check-natives.py smoke-set <t>    print the natives jar classifiers of smoke target <t>, one per line
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
    if argv[1:] == ["smoke-targets"]:
        print("\n".join(natives.smoke_targets(rows)))
        return 0
    if len(argv) == 3 and argv[1] == "smoke-set":
        if argv[2] not in natives.smoke_targets(rows):
            print(f"{argv[2]} is not a smoke target: {natives.smoke_targets(rows)}", file=sys.stderr)
            return 2
        print("\n".join(r["classifier"] for r in natives.smoke_set(rows, argv[2])))
        return 0
    if argv[1:]:
        print(__doc__, file=sys.stderr)
        return 2
    failures = natives.check(ROOT) + models.check(natives.read(ROOT, ".github/models.csv"),
                                                  natives.read(ROOT, ".github/workflows/publish.yml"))
    for f in failures:
        print(f"::error::{f}", file=sys.stderr)
    print(f"{len(rows)} natives jars ({len(natives.libraries(rows))} library, {len(natives.modules(rows))} module), "
          f"{len(natives.smoke_targets(rows))} smoke targets, {len(failures)} disagreements")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
