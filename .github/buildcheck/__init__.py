# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""This repository's own build checks: the natives list, the patches, the native dependencies, the
HIP offload bundles, the WebUI license texts, the AAR poms. Standard library only.

The checks it shares with its sibling repositories live in bernardladenthin/actions (actionlib),
and some of the modules here build on it (natives.py reads the job graph with its workflow parser).
actionlib is found on PYTHONPATH (common/setup in CI), else in ACTIONS_DIR, else in a checkout
of bernardladenthin/actions next to this repository (../actions); this module adds the last two
to sys.path when actionlib is not importable already.

Run the tests from the repository root:
  python3 -m unittest discover -s .github/buildcheck/tests -t .github
"""

import importlib.util
import os
import sys

if importlib.util.find_spec("actionlib") is None:
    _repository = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for _candidate in (os.environ.get("ACTIONS_DIR"), os.path.join(os.path.dirname(_repository), "actions")):
        if _candidate and os.path.isfile(os.path.join(_candidate, "lib", "actionlib", "__init__.py")):
            sys.path.insert(0, os.path.join(_candidate, "lib"))
            break
