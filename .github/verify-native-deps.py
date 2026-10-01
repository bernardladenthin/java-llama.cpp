#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Fail when a shipped native library needs a runtime library it did not need before.

See buildcheck/nativedeps.py. Usage: verify-native-deps.py <natives-root>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from buildcheck import nativedeps  # noqa: E402

if __name__ == "__main__":
    sys.exit(nativedeps.main(sys.argv))
