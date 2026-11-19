#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate or check the license texts of the npm packages in the embedded WebUI.

  webui-licenses.py generate <llama.cpp>/ui/apps/chat llama/legal/natives/webui-npm
  webui-licenses.py check <llama.cpp>/ui/apps/chat llama/legal/natives/webui-npm
"""

import argparse
import sys

from buildcheck.webuilicenses import check, check_notice, check_reuse, generate


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("generate", "check"))
    parser.add_argument("ui", help="llama.cpp's ui/apps/chat directory at the pinned commit")
    parser.add_argument("out", help="llama/legal/natives/webui-npm")
    args = parser.parse_args()
    if args.command == "generate":
        packages = generate(args.ui, args.out)
        print(f"{len(packages)} packages written to {args.out}")
        return 0
    problems = check(args.ui, args.out) + check_reuse(args.out) + check_notice(args.out)
    for problem in problems:
        print(f"::error::{problem}")
    if not problems:
        print("WebUI license texts match ui/apps/chat/package-lock.json")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
