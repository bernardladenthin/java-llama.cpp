#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Usage: collect-release-assets.py <destination>; run from the repository root."""

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from buildcheck.releaseassets import collect


def main():
    errors = []
    try:
        if len(sys.argv) != 2:
            raise ValueError(__doc__)
        errors = collect(Path(__file__).resolve().parent.parent, sys.argv[1])
    except (ValueError, OSError, ET.ParseError) as error:
        errors.append(str(error))
    for error in errors:
        print(f"::error::{error}")
    if not errors:
        print(f"Collected {len(list(Path(sys.argv[1]).iterdir()))} release files including signatures.")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write(f"complete={'false' if errors else 'true'}\n")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
