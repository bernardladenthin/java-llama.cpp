#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""This repository's own post-release checks of a published release.

See buildcheck/postrelease.py. Usage: post-release-check.py --tag v<version> --assets <dir> [--checks ...]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from buildcheck import postrelease  # noqa: E402

if __name__ == "__main__":
    sys.exit(postrelease.main(sys.argv))
