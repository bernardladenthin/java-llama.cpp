#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Fail when a published Android AAR pom does not mirror the core's compile-scope dependencies.

See buildcheck/aarpoms.py. Usage: verify-aar-poms.py <repo-root> <aar-pom>...
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from buildcheck import aarpoms  # noqa: E402

if __name__ == "__main__":
    sys.exit(aarpoms.main(sys.argv))
