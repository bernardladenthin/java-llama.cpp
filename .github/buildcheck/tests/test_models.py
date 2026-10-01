# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

import unittest

from buildcheck import models, natives
from buildcheck.tests.helpers import REPO

CSV = "# comment\na.gguf,https://x/a\nb.gguf,https://x/b\n"


class ModelsTest(unittest.TestCase):

    def test_env_names_must_be_listed(self):
        workflow = 'env:\n  JAVA_VERSION: \'21\'\n  A_MODEL_NAME: "a.gguf"\n  B: b.gguf\n'
        self.assertEqual(models.check(CSV, workflow), [])
        self.assertEqual(models.check(CSV, workflow + '  C_MODEL_NAME: "c.gguf"\n'),
                         ["publish.yml env C_MODEL_NAME=c.gguf is not a filename of .github/models.csv "
                          "-- no job downloads it"])

    def test_the_repository_passes(self):
        self.assertEqual(models.check(natives.read(REPO, ".github/models.csv"),
                                      natives.read(REPO, ".github/workflows/publish.yml")), [])


if __name__ == "__main__":
    unittest.main()
