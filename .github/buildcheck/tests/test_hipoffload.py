# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT

import os
import tempfile
import unittest

from buildcheck import hipoffload


class HipOffloadTest(unittest.TestCase):

    def library(self, folder, content):
        path = os.path.join(folder, "libjllama.so")
        with open(path, "wb") as f:
            f.write(content)
        return path

    def test_counts_a_magic_split_across_two_chunks_once(self):
        with tempfile.TemporaryDirectory() as d:
            magic = hipoffload.UNCOMPRESSED
            path = self.library(d, b"x" * 30 + magic + b"y" * 5 + hipoffload.COMPRESSED)
            for chunk in (7, 32, 40, 1 << 20):
                with self.subTest(chunk=chunk):
                    self.assertEqual(hipoffload.count(path, chunk), (1, 1))

    def test_exit_codes(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(hipoffload.main(["x", d]), 2)  # nothing found
            path = self.library(d, b"CCOB..CCOB")
            self.assertEqual(hipoffload.main(["x", d]), 0)
            self.library(d, b"CCOB" + hipoffload.UNCOMPRESSED)
            self.assertEqual(hipoffload.main(["x", path]), 1)  # a file argument works as well
            self.library(d, b"no bundles")
            self.assertEqual(hipoffload.main(["x", d]), 1)
        self.assertEqual(hipoffload.main(["x"]), 2)


if __name__ == "__main__":
    unittest.main()
