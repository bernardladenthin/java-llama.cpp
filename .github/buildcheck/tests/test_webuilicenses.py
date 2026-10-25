# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The WebUI npm license texts: closure over the lockfile, texts from tarballs, the staleness check."""

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from buildcheck import webuilicenses


def tarball(files, root="package"):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        for name, content in files.items():
            raw = content.encode("utf-8")
            info = tarfile.TarInfo(f"{root}/{name}")
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    return data.getvalue()


class WebuiLicensesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ui = Path(self.temp.name) / "ui"
        (self.ui / "src/lib").mkdir(parents=True)
        (self.ui / "src/lib/app.ts").write_text(
            "import { a } from 'alpha';\nimport type { T } from 'typesonly';\nimport './local';\n")
        (self.ui / "src/lib/app.test.ts").write_text("import { t } from 'testonly';\n")
        lock = {"packages": {
            "": {},
            "node_modules/alpha": {"version": "1.0.0", "license": "MIT", "dependencies": {"beta": "^2"},
                                   "resolved": "https://registry.npmjs.org/alpha/-/alpha-1.0.0.tgz"},
            "node_modules/beta": {"version": "2.0.0", "license": "ISC", "dependencies": {"@types/x": "*"},
                                  "resolved": "https://registry.npmjs.org/beta/-/beta-2.0.0.tgz"},
            "node_modules/alpha/node_modules/beta": {"version": "2.0.0", "license": "ISC",
                                                     "resolved": "https://registry.npmjs.org/beta/-/beta-2.0.0.tgz"},
            "node_modules/@types/x": {"version": "1.0.0", "license": "MIT",
                                      "resolved": "https://registry.npmjs.org/@types/x/-/x-1.0.0.tgz"},
            "node_modules/typesonly": {"version": "1.0.0", "resolved": "https://registry.npmjs.org/t.tgz"},
            "node_modules/testonly": {"version": "1.0.0", "resolved": "https://registry.npmjs.org/t.tgz"},
            "node_modules/svelte": {"version": "5.0.0", "license": "MIT",
                                    "resolved": "https://registry.npmjs.org/svelte/-/svelte-5.0.0.tgz"},
        }}
        (self.ui / "package-lock.json").write_text(json.dumps(lock))
        self.out = Path(self.temp.name) / "out"

    def test_closure_follows_runtime_imports_and_drops_types_tests_and_duplicates(self):
        lock = json.loads((self.ui / "package-lock.json").read_text())
        names = [p["name"] for p in webuilicenses.closure(lock, webuilicenses.imports(self.ui))]
        self.assertEqual(names, ["alpha", "beta", "svelte"])

    def test_generate_writes_the_tarball_texts_and_the_declared_license_of_the_rest(self):
        tarballs = {
            "https://registry.npmjs.org/alpha/-/alpha-1.0.0.tgz": tarball({"LICENSE": "alpha license",
                                                                           "lib/LICENSE": "nested"}),
            "https://registry.npmjs.org/beta/-/beta-2.0.0.tgz": tarball(
                {"package.json": json.dumps({"author": {"name": "Beta Author"}})}),
            "https://registry.npmjs.org/svelte/-/svelte-5.0.0.tgz": tarball({"LICENSE.md": "svelte"}, root="svelte"),
        }
        webuilicenses.generate(self.ui, self.out, fetch=tarballs.__getitem__, workers=1)
        text = (self.out / "THIRD-PARTY-LICENSES.txt").read_text(encoding="utf-8")
        self.assertIn("alpha@1.0.0 (MIT): LICENSE\n", text)
        self.assertIn("alpha license", text)
        self.assertNotIn("nested", text)
        self.assertIn("License (package.json): ISC", text)
        self.assertIn("Author (package.json): Beta Author", text)
        self.assertIn("svelte@5.0.0 (MIT): LICENSE.md", text)
        self.assertEqual(webuilicenses.check(self.ui, self.out), [])

    def test_check_reports_a_changed_lockfile(self):
        self.out.mkdir()
        (self.out / "LOCKFILE.sha256").write_text("0" * 64 + "  tools/ui/package-lock.json\n")
        problems = webuilicenses.check(self.ui, self.out)
        self.assertEqual(len(problems), 1)
        self.assertIn("webui-licenses.py generate", problems[0])


if __name__ == "__main__":
    unittest.main()
