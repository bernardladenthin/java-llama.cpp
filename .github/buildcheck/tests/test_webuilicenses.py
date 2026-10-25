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
        # beta ships no license file: the full ISC text follows, once.
        isc = (webuilicenses.SPDX_TEXTS / "ISC.txt").read_text(encoding="utf-8").strip()
        self.assertIn("License text ISC (for the packages above marked (declared in package.json))", text)
        self.assertEqual(text.count(isc.splitlines()[-1]), 1)
        self.assertNotIn("License text MIT", text)
        self.assertEqual(webuilicenses.check(self.ui, self.out), [])

    def generated(self):
        tarballs = {
            "https://registry.npmjs.org/alpha/-/alpha-1.0.0.tgz": tarball({"LICENSE": "alpha license"}),
            "https://registry.npmjs.org/beta/-/beta-2.0.0.tgz": tarball({"LICENSE": "beta license"}),
            "https://registry.npmjs.org/svelte/-/svelte-5.0.0.tgz": tarball({"LICENSE": "svelte"}),
            "https://registry.npmjs.org/t.tgz": tarball({"LICENSE": "t"}),
        }
        webuilicenses.generate(self.ui, self.out, fetch=tarballs.__getitem__, workers=1)
        self.assertEqual(webuilicenses.check(self.ui, self.out), [])

    def test_check_reports_a_changed_lockfile(self):
        self.generated()
        (self.out / "LOCKFILE.sha256").write_text("0" * 64 + "  tools/ui/package-lock.json\n")
        problems = webuilicenses.check(self.ui, self.out)
        self.assertEqual(len(problems), 1)
        self.assertIn("webui-licenses.py generate", problems[0])

    def test_the_lockfile_digest_does_not_depend_on_the_line_ends(self):
        lock = self.ui / "package-lock.json"
        lf = (json.dumps(json.loads(lock.read_text(encoding="utf-8")), indent=2) + "\n").encode("utf-8")
        lock.write_bytes(lf)
        digest = webuilicenses.lockfile_digest(self.ui)
        lock.write_bytes(lf.replace(b"\n", b"\r\n"))
        self.assertEqual(webuilicenses.lockfile_digest(self.ui), digest)

    def test_check_reports_a_new_import_of_a_package_the_lockfile_already_had(self):
        self.generated()
        (self.ui / "src/lib/more.ts").write_text("import { t } from 'testonly';\n")
        problems = webuilicenses.check(self.ui, self.out)
        self.assertEqual(len(problems), 1)
        self.assertIn("1 package(s) the WebUI sources can reach are not in packages.csv (testonly@1.0.0)",
                      problems[0])

    def test_check_reports_a_listed_file_without_its_section(self):
        self.generated()
        path = self.out / "THIRD-PARTY-LICENSES.txt"
        path.write_text(path.read_text(encoding="utf-8").replace("beta@2.0.0 (ISC): LICENSE", "beta"),
                        encoding="utf-8")
        self.assertEqual(webuilicenses.check(self.ui, self.out),
                         ["beta@2.0.0: no section for LICENSE in THIRD-PARTY-LICENSES.txt"])

    def test_check_reports_a_declared_only_license_without_its_text(self):
        tarballs = {
            "https://registry.npmjs.org/alpha/-/alpha-1.0.0.tgz": tarball({"LICENSE": "alpha license"}),
            "https://registry.npmjs.org/beta/-/beta-2.0.0.tgz": tarball({"index.js": ""}),
            "https://registry.npmjs.org/svelte/-/svelte-5.0.0.tgz": tarball({"LICENSE": "svelte"}),
        }
        webuilicenses.generate(self.ui, self.out, fetch=tarballs.__getitem__, workers=1)
        path = self.out / "THIRD-PARTY-LICENSES.txt"
        text = path.read_text(encoding="utf-8")
        path.write_text(text[:text.index("\n" + "=" * 78 + "\nLicense text ISC")] + "\n", encoding="utf-8")
        self.assertEqual(webuilicenses.check(self.ui, self.out),
                         ["no ISC license text in THIRD-PARTY-LICENSES.txt for the packages that only declare it"])

    def test_generate_refuses_a_package_without_a_license_file_or_a_declared_license(self):
        lock = json.loads((self.ui / "package-lock.json").read_text())
        for key in ("node_modules/beta", "node_modules/alpha/node_modules/beta"):
            del lock["packages"][key]["license"]
        (self.ui / "package-lock.json").write_text(json.dumps(lock))
        tarballs = {
            "https://registry.npmjs.org/alpha/-/alpha-1.0.0.tgz": tarball({"LICENSE": "alpha license"}),
            "https://registry.npmjs.org/beta/-/beta-2.0.0.tgz": tarball({"index.js": ""}),
            "https://registry.npmjs.org/svelte/-/svelte-5.0.0.tgz": tarball({"LICENSE": "svelte"}),
        }
        with self.assertRaisesRegex(ValueError, "declares no license: beta@2.0.0"):
            webuilicenses.generate(self.ui, self.out, fetch=tarballs.__getitem__, workers=1)
        self.assertFalse(self.out.exists())

    def test_declared_ids_leave_the_operators_out(self):
        self.assertEqual(webuilicenses.declared_ids("(MPL-2.0 OR Apache-2.0)"), ["MPL-2.0", "Apache-2.0"])
        self.assertEqual(webuilicenses.declared_ids(""), [])


if __name__ == "__main__":
    unittest.main()
