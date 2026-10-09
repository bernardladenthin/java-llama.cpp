#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Fail when a patch declares a hunk size that does not match its body.

See buildcheck/patches.py -- `git apply --check` does not catch this and a truncated file is a
successful apply as far as git is concerned. Usage: check-patches.py [<patch-dir>]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from buildcheck import patches  # noqa: E402

if __name__ == "__main__":
    sys.exit(patches.main(sys.argv))
