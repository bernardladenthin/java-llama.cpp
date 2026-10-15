# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

import os
import unittest

from buildcheck import patches
from buildcheck.tests.helpers import REPO


def new_file_patch(body, declared=None):
    """A git patch adding `new.h` with the given lines; `declared` overrides the hunk's +count."""
    n = len(body) if declared is None else declared
    return ("some prose above the diff, as the project's patch headers carry\n"
            "diff --git a/dir/new.h b/dir/new.h\n"
            "new file mode 100644\n"
            "index 0000000..1234567\n"
            "--- /dev/null\n"
            "+++ b/dir/new.h\n"
            f"@@ -0,0 +1,{n} @@\n" + "".join(f"+{line}\n" for line in body))


MODIFY = ("diff --git a/a.c b/a.c\n"
          "index 1111111..2222222 100644\n"
          "--- a/a.c\n"
          "+++ b/a.c\n"
          "@@ -1,4 +1,5 @@\n"
          " one\n"
          "-two\n"
          "+two changed\n"
          "+two and a half\n"
          " three\n"
          " four\n"
          "\\ No newline at end of file\n")


class AuditTest(unittest.TestCase):

    def test_a_new_file_hunk_whose_header_matches_its_body_passes(self):
        problems, hunks = patches.audit(new_file_patch(["a", "b", "c"]))
        self.assertEqual(problems, [])
        self.assertEqual(hunks, 1)

    def test_lines_past_the_declared_count_are_reported_the_489_shape(self):
        """Two comment lines were added to prefetch.h without recounting: 27 lines under a +25
        header. `git apply` writes 25 and drops the rest without a word."""
        problems, _ = patches.audit(new_file_patch(["a", "b", "c", "d", "e"], declared=3), "p.patch")
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("p.patch:7", problems[0])
        self.assertIn("2 line(s) after the hunk for dir/new.h", problems[0])
        self.assertIn("-0,+3", problems[0])

    def test_a_header_claiming_more_than_the_body_carries_is_reported(self):
        problems, _ = patches.audit(new_file_patch(["a", "b"], declared=4), "p.patch")
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("declares -0,+4 lines but its body ends after -0,+2", problems[0])

    def test_a_modify_hunk_with_context_removals_additions_and_the_no_newline_marker_passes(self):
        self.assertEqual(patches.audit(MODIFY), ([], 1))

    def test_an_excess_removed_or_context_line_counts_as_well(self):
        for extra in ("-gone\n", " ctx\n"):
            problems, _ = patches.audit(MODIFY + extra, "p.patch")
            self.assertEqual(len(problems), 1, (extra, problems))
            self.assertIn("1 line(s) after the hunk for a.c", problems[0])

    def test_the_next_file_header_and_the_format_patch_trailer_end_a_hunk_cleanly(self):
        two = new_file_patch(["a"]) + new_file_patch(["b", "c"]).split("\n", 1)[1] + "-- \n2.43.0\n\n"
        problems, hunks = patches.audit(two)
        self.assertEqual(problems, [])
        self.assertEqual(hunks, 2)

    def test_the_prose_before_the_first_diff_is_not_inspected(self):
        text = "+ a plus sign in prose\n- and a dash\n" + new_file_patch(["x"])
        self.assertEqual(patches.audit(text), ([], 1))


class RepositoryTest(unittest.TestCase):

    def test_every_patch_of_this_repository_is_consistent(self):
        problems, count, hunks = patches.check(REPO)
        self.assertEqual(problems, [])
        self.assertGreaterEqual(count, 11)
        self.assertGreater(hunks, count)

    def test_the_broken_0017_of_pr_489_is_caught(self):
        """The patch text as merged in #489 (main 4b32dae5..bc2c8af9): header +1,25 over 27 lines."""
        import subprocess
        try:
            text = subprocess.run(["git", "-C", REPO, "show",
                                   "bc2c8af9:llama/patches/0017-ggml-cpu-x86-prefetch-byte-offset-and-msvc-type.patch"],
                                  capture_output=True, check=True).stdout.decode("utf-8", errors="surrogateescape")
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.skipTest("the historic commit is not available in this checkout")
        problems, _ = patches.audit(text, "0017.patch")
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("2 line(s) after the hunk for ggml/src/ggml-cpu/arch/x86/prefetch.h", problems[0])
        self.assertIn("-0,+25", problems[0])


class RenderTest(unittest.TestCase):
    """docs/patches.md is generated from the patch headers; the check fails when it is stale."""

    def test_describe_reads_title_prose_files_and_hunks(self):
        text = ("ggml-cpu: fix the thing\n\nTwo paragraphs of why.\n" + MODIFY
                + MODIFY.replace("a.c", "b/dir/c.h").replace("b/b/dir", "b/dir"))
        e = patches.describe(text, "0042-fix-the-thing.patch")
        self.assertEqual(e["id"], "0042")
        self.assertEqual(e["title"], "ggml-cpu: fix the thing")
        self.assertTrue(e["prose"].startswith("ggml-cpu: fix the thing"))
        self.assertEqual(e["files"], ["a.c", "dir/c.h"])
        self.assertEqual(e["hunks"], 2)

    def test_a_patch_without_prose_is_titled_by_its_file_name(self):
        e = patches.describe(MODIFY, "0001-no-prose.patch")
        self.assertEqual(e["title"], "0001-no-prose.patch")
        self.assertEqual(e["prose"], "")

    def test_render_lists_every_patch_once_and_escapes_the_table(self):
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            d = os.path.join(root, "llama", "patches")
            os.makedirs(d)
            with open(os.path.join(d, "0002-second.patch"), "w") as f:
                f.write("second | with a pipe\n" + MODIFY)
            with open(os.path.join(d, "0001-first.patch"), "w") as f:
                f.write("first\n\n```\nfenced\n```\n" + MODIFY)
            page = patches.render(root)
        self.assertIn("**2 patches.**", page)
        self.assertLess(page.index("## 0001-first.patch"), page.index("## 0002-second.patch"))
        self.assertIn("| 0002 | [`0002-second.patch`](#0002-secondpatch) | second \\| with a pipe | 1 | 1 |", page)
        self.assertNotIn("\n```\nfenced", page, "a fence inside the prose must not close the code block")
        self.assertEqual(page.count("```text"), 2)

    def test_doc_is_current_compares_the_committed_page_with_the_render(self):
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            d = os.path.join(root, "llama", "patches")
            os.makedirs(d)
            with open(os.path.join(d, "0001-first.patch"), "w") as f:
                f.write("first\n" + MODIFY)
            self.assertEqual(patches.doc_is_current(root)[0], False, "no page yet")
            os.makedirs(os.path.join(root, "docs"))
            with open(os.path.join(root, "docs", "patches.md"), "w") as f:
                f.write(patches.render(root))
            self.assertTrue(patches.doc_is_current(root)[0])
            with open(os.path.join(root, "docs", "patches.md"), "a") as f:
                f.write("hand edit\n")
            self.assertFalse(patches.doc_is_current(root)[0])

    def test_the_committed_page_of_this_repository_is_current(self):
        current, path = patches.doc_is_current(REPO)
        self.assertTrue(current, f"{path} is stale: run python3 .github/check-patches.py --render")
