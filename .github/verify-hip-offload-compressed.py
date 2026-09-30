#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Fail when a ROCm/HIP build ships its GPU code uncompressed.

See buildcheck/hipoffload.py. Usage: verify-hip-offload-compressed.py <dir-or-file>...
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from buildcheck import hipoffload  # noqa: E402

if __name__ == "__main__":
    sys.exit(hipoffload.main(sys.argv))
