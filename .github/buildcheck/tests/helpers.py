# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Fixtures of this repository's build-check tests; the workflow fixtures come from actionlib."""

import os

import buildcheck  # noqa: F401 -- finds actionlib (see buildcheck/__init__.py)
from actionlib.testing import upload, workflow_text  # noqa: F401 -- re-exported for the tests

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
