# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

import unittest

from buildcheck import models, natives
from buildcheck.tests.helpers import REPO

COMMIT = "0123456789abcdef0123456789abcdef01234567"
SHA = "ab" * 32
CSV = ("# comment\n"
       f"a.gguf,draft,{SHA},https://huggingface.co/o/r/resolve/{COMMIT}/a.gguf\n"
       f"b.gguf,tts,{SHA},https://huggingface.co/o/r/resolve/{COMMIT}/sub/b.gguf\n")


class ModelsTest(unittest.TestCase):

    def test_env_names_must_be_listed(self):
        workflow = 'env:\n  JAVA_VERSION: \'21\'\n  A_MODEL_NAME: "a.gguf"\n  B: b.gguf\n'
        self.assertEqual(models.check(CSV, workflow), [])
        self.assertEqual(models.check(CSV, workflow + '  C_MODEL_NAME: "c.gguf"\n'),
                         ["publish.yml env C_MODEL_NAME=c.gguf is not a filename of .github/models.csv "
                          "-- no job downloads it"])

    def test_requested_features_must_be_listed(self):
        workflow = "      - uses: ./.github/actions/fetch-models\n        with:\n          features: draft tts\n"
        self.assertEqual(models.check(CSV, workflow), [])
        self.assertEqual(models.check(CSV, workflow.replace("tts", "tts vison")),
                         ["publish.yml asks fetch-models for the feature 'vison', which no row of "
                          ".github/models.csv has"])

    def test_a_row_has_four_fields(self):
        self.assertEqual(models.check_rows("a.gguf,https://huggingface.co/o/r/resolve/main/a.gguf\n"),
                         ["models.csv row 'a.gguf,https://huggingface.co/o/r/resolve/main/a.gguf' is not "
                          "filename,feature,sha256,url"])

    def test_the_url_is_pinned_to_a_commit_and_names_the_file(self):
        unpinned = CSV.replace(f"resolve/{COMMIT}/a.gguf", "resolve/main/a.gguf")
        self.assertEqual(len(models.check_rows(unpinned)), 1)
        self.assertIn("pinned to a commit", models.check_rows(unpinned)[0])
        other = CSV.replace(f"{COMMIT}/a.gguf", f"{COMMIT}/c.gguf")
        self.assertEqual(models.check_rows(other), ["models.csv: a.gguf: the URL names another file, c.gguf"])

    def test_hash_feature_and_duplicates(self):
        self.assertEqual(models.check_rows(CSV.replace(SHA, "AB" * 32, 1)),
                         [f"models.csv: a.gguf: {'AB' * 32!r} is no lower-case SHA-256"])
        self.assertEqual(models.check_rows(CSV.replace(",draft,", ",Draft model,")),
                         ["models.csv: a.gguf: feature 'Draft model' is not one lower-case word"])
        first = CSV.splitlines()[1]
        self.assertEqual(models.check_rows(CSV + first + "\n"), ["models.csv: a.gguf is listed twice"])

    def test_the_repository_passes(self):
        self.assertEqual(models.check(natives.read(REPO, ".github/models.csv"),
                                      natives.read(REPO, ".github/workflows/publish.yml")), [])


if __name__ == "__main__":
    unittest.main()
