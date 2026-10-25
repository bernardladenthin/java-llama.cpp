#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate or check the license texts of the npm packages in the embedded WebUI.

  webui-licenses.py generate <llama.cpp>/tools/ui llama/legal/natives/webui-npm
  webui-licenses.py check <llama.cpp>/tools/ui llama/legal/natives/webui-npm [--warn-only]
"""

import argparse
import sys

from buildcheck.webuilicenses import check, generate


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("generate", "check"))
    parser.add_argument("ui", help="llama.cpp's tools/ui directory at the pinned commit")
    parser.add_argument("out", help="llama/legal/natives/webui-npm")
    parser.add_argument("--warn-only", action="store_true", help="report a stale list as a warning")
    args = parser.parse_args()
    if args.command == "generate":
        packages = generate(args.ui, args.out)
        print(f"{len(packages)} packages written to {args.out}")
        return 0
    problems = check(args.ui, args.out)
    for problem in problems:
        print(f"::{'warning' if args.warn_only else 'error'}::{problem}")
    if not problems:
        print("WebUI license texts match tools/ui/package-lock.json")
    return 1 if problems and not args.warn_only else 0


if __name__ == "__main__":
    sys.exit(main())
