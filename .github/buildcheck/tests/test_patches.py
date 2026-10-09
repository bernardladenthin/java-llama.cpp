# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Tests for buildcheck.patches -- the hunk-size guard."""

import io
import os
import tempfile
import unittest

from buildcheck import patches

# A correct new-file hunk: 4 added lines, declared as such.
GOOD_NEW_FILE = """diff --git a/x.h b/x.h
new file mode 100644
index 0000000..1111111
--- /dev/null
+++ b/x.h
@@ -0,0 +1,4 @@
+#pragma once
+static inline int f(void) {
+    return 1;
+}
"""

# A correct modification hunk: 3 context + 1 removed + 1 added = -4 +4.
GOOD_EDIT = """diff --git a/y.c b/y.c
--- a/y.c
+++ b/y.c
@@ -10,4 +10,4 @@ int main(void)
 a;
-old;
+new;
 b;
 c;
"""


class HunkSizeTest(unittest.TestCase):
    def write(self, text):
        path = os.path.join(self.dir, "p.patch")
        with io.open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        return path

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def test_a_correct_new_file_hunk_passes(self):
        self.assertEqual(patches.problems(self.write(GOOD_NEW_FILE)), [])

    def test_a_correct_modification_hunk_passes(self):
        self.assertEqual(patches.problems(self.write(GOOD_EDIT)), [])

    def test_an_under_counting_header_is_reported(self):
        """The regression this guard exists for: the body is longer than the header says, so
        `git apply` writes only the declared lines and truncates the file -- and --check is happy."""
        broken = GOOD_NEW_FILE.replace("@@ -0,0 +1,4 @@", "@@ -0,0 +1,2 @@")
        found = patches.problems(self.write(broken))
        self.assertEqual(len(found), 1, found)
        self.assertIn("UNDER-counts", found[0])
        self.assertIn("TRUNCATED", found[0])
        # it names the line the body continues at, which is the first line git would drop
        self.assertIn("return 1", found[0])

    def test_an_over_counting_header_is_reported(self):
        over = GOOD_NEW_FILE.replace("@@ -0,0 +1,4 @@", "@@ -0,0 +1,9 @@")
        found = patches.problems(self.write(over))
        self.assertEqual(len(found), 1, found)
        self.assertIn("SHORT", found[0])

    def test_a_wrong_old_side_count_is_reported(self):
        wrong = GOOD_EDIT.replace("@@ -10,4 +10,4 @@", "@@ -10,9 +10,4 @@")
        found = patches.problems(self.write(wrong))
        self.assertEqual(len(found), 1, found)
        self.assertIn("SHORT", found[0])

    def test_a_single_line_hunk_may_omit_its_count(self):
        """`@@ -5 +5 @@` is legal shorthand for a count of 1 and must not be flagged."""
        one = "--- a/z\n+++ b/z\n@@ -5 +5 @@\n-a\n+b\n"
        self.assertEqual(patches.problems(self.write(one)), [])

    def test_a_no_newline_marker_counts_for_neither_side(self):
        marker = ("--- a/z\n+++ b/z\n@@ -1,2 +1,2 @@\n a\n-b\n" + chr(92) +
                  " No newline at end of file\n+c\n" + chr(92) + " No newline at end of file\n")
        self.assertEqual(patches.problems(self.write(marker)), [])

    def test_prose_after_the_last_hunk_does_not_count(self):
        """The patches in this repository carry a long description ABOVE the diff, and some end
        with a git signature; neither may be read as hunk body."""
        self.assertEqual(patches.problems(self.write(GOOD_EDIT + "-- \n2.43.0\n")), [])

    def test_every_hunk_of_a_multi_hunk_patch_is_checked(self):
        two = GOOD_EDIT + GOOD_NEW_FILE.replace("@@ -0,0 +1,4 @@", "@@ -0,0 +1,3 @@")
        found = patches.problems(self.write(two))
        self.assertEqual(len(found), 1, found)
        self.assertIn("UNDER-counts", found[0])


class MainTest(unittest.TestCase):
    def test_the_repositorys_own_patches_are_consistent(self):
        here = os.path.dirname(os.path.abspath(__file__))
        repo = os.path.dirname(os.path.dirname(os.path.dirname(here)))
        self.assertEqual(patches.main(["x", os.path.join(repo, "llama", "patches")]), 0)

    def test_no_patches_found_is_a_failure(self):
        with tempfile.TemporaryDirectory() as empty:
            self.assertEqual(patches.main(["x", empty]), 2)


if __name__ == "__main__":
    unittest.main()
