# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from buildcheck.releaseassets import collect
from buildcheck import workflow
from buildcheck.tests.helpers import REPO


VERSION = "6.0.0-SNAPSHOT"


class ReleaseAssetsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "assets"
        self.write("pom.xml", f"""<project xmlns="http://maven.apache.org/POM/4.0.0">
<artifactId>llama-parent</artifactId><version>{VERSION}</version><packaging>pom</packaging>
<modules><module>llama</module><module>llama-ffm</module><module>llama-platform</module></modules>
</project>""")
        self.outputs(".", "llama-parent", "pom")
        for name, packaging in (("llama", "jar"), ("llama-ffm", "jar"), ("llama-platform", "pom")):
            self.module(name, packaging)
        self.module("llama-atmosphere-agent", "jar", reactor=False)
        self.write(".github/natives.csv", "# comment\nclassifier,directory\ncpu-linux-x86-64,Linux/x86_64/cpu\n")
        self.asset(f"llama/target/llama-{VERSION}-cpu-linux-x86-64.jar")
        for artifact, publication in (("llama-android", "llamaAndroid"),
                                      ("llama-android-opencl", "llamaAndroidOpencl")):
            self.asset(f"llama-android/build/aar/{artifact}-{VERSION}.aar")
            self.asset(f"llama-android/build/publications/{publication}/pom-default.xml")
        for classifier in ("sources", "javadoc"):
            self.asset(f"llama-android/build/distributions/llama-android-{VERSION}-{classifier}.jar")
        self.asset(f"llama-source-archive/llama.cpp-b11827-jllama-{VERSION}-src.tar.gz")

    def write(self, name, content="artifact"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def asset(self, name):
        self.write(name)
        self.write(name + ".asc", "signature of " + name)

    def outputs(self, folder, artifact, packaging, reactor=True):
        self.asset(f"{folder}/target/{artifact}-{VERSION}.pom")
        if packaging == "jar":
            for classifier in ("", "-sources", "-javadoc"):
                self.asset(f"{folder}/target/{artifact}-{VERSION}{classifier}.jar")
        if reactor:
            self.asset(f"{folder}/target/bom.json")

    def module(self, name, packaging, reactor=True):
        self.write(f"{name}/pom.xml", f"""<project>
<parent><version>{VERSION}</version></parent><artifactId>{name}</artifactId>
<packaging>{packaging}</packaging></project>""")
        self.outputs(name, name, packaging, reactor)

    def test_collects_ffm_and_renames_each_bom_and_signature_to_its_central_name(self):
        self.assertEqual(collect(self.root, self.output), [])
        for classifier in ("", "-sources", "-javadoc"):
            self.assertTrue((self.output / f"llama-ffm-{VERSION}{classifier}.jar.asc").is_file())
        for name in ("llama-parent", "llama", "llama-ffm", "llama-platform"):
            self.assertEqual((self.output / f"{name}-{VERSION}-cyclonedx.json").read_text(), "artifact")
            self.assertIn(f"{name if name != 'llama-parent' else '.'}/target/bom.json",
                          (self.output / f"{name}-{VERSION}-cyclonedx.json.asc").read_text())
        self.assertFalse((self.output / "bom.json").exists())
        self.assertTrue((self.output / f"llama-android-opencl-{VERSION}-sources.jar.asc").is_file())

    def test_new_reactor_module_is_included_without_another_collector_list(self):
        pom = self.root / "pom.xml"
        pom.write_text(pom.read_text().replace("</modules>", "<module>new-module</module></modules>"))
        self.module("new-module", "jar")
        self.assertEqual(collect(self.root, self.output), [])
        self.assertTrue((self.output / f"new-module-{VERSION}.jar").is_file())
        self.assertTrue((self.output / f"new-module-{VERSION}-cyclonedx.json").is_file())

    def test_missing_ffm_is_reported_but_available_artifacts_survive(self):
        path = self.root / f"llama-ffm/target/llama-ffm-{VERSION}.jar"
        path.unlink()
        errors = collect(self.root, self.output)
        self.assertTrue(any(str(path) in error for error in errors))
        self.assertTrue((self.output / f"llama-{VERSION}.jar").is_file())

    def test_missing_signatures_boms_natives_and_aars_are_not_silently_ignored(self):
        missing = ["llama/target/bom.json", f"llama-ffm/target/llama-ffm-{VERSION}.pom.asc",
                   f"llama/target/llama-{VERSION}-cpu-linux-x86-64.jar",
                   f"llama-android/build/aar/llama-android-opencl-{VERSION}.aar"]
        for name in missing:
            (self.root / name).unlink()
        errors = collect(self.root, self.output)
        for name in missing:
            self.assertTrue(any(str(self.root / name) in error for error in errors))

    def test_stale_versions_and_fat_jars_are_not_collected(self):
        self.asset("llama/target/llama-5.2.0-SNAPSHOT.jar")
        self.asset(f"llama/target/llama-{VERSION}-jar-with-dependencies.jar")
        self.assertEqual(collect(self.root, self.output), [])
        self.assertFalse(any("5.2.0" in path.name or "jar-with-dependencies" in path.name
                             for path in self.output.iterdir()))

    def test_nonempty_destination_and_version_mismatch_fail(self):
        self.write("assets/stale.jar")
        with self.assertRaisesRegex(ValueError, "must be empty"):
            collect(self.root, self.output)
        (self.output / "stale.jar").unlink()
        pom = self.root / "llama-ffm/pom.xml"
        pom.write_text(pom.read_text().replace(VERSION, "5.2.0-SNAPSHOT"))
        with self.assertRaisesRegex(ValueError, "version mismatch"):
            collect(self.root, self.output)

    def test_multiple_source_archives_are_reported(self):
        self.asset(f"llama-source-archive/llama.cpp-b11820-jllama-{VERSION}-src.tar.gz")
        self.assertTrue(any("found 2" in error for error in collect(self.root, self.output)))

    def test_both_publish_paths_collect_and_attest_json_and_report_signature_gaps(self):
        jobs = workflow.parse((Path(REPO) / ".github/workflows/publish.yml").read_text())
        for kind in ("snapshot", "release"):
            job = jobs[f"publish-{kind}"].text
            self.assertIn(f"python3 .github/collect-release-assets.py signed-{kind}-assets", job)
            self.assertIn(f"signed-{kind}-assets/*.json", job)
            self.assertIn("assets_complete: ${{ steps.collect.outputs.complete }}", job)
            attach = jobs["github-snapshot" if kind == "snapshot" else "github-release-signed"].text
            self.assertIn("-name '*.json'", attach)

    def test_cli_reports_completeness_even_when_collection_fails(self):
        scripts = self.root / ".github"
        (scripts / "buildcheck").mkdir()
        for name in ("collect-release-assets.py", "buildcheck/releaseassets.py", "buildcheck/__init__.py"):
            shutil.copyfile(Path(REPO) / ".github" / name, scripts / name)
        for complete in (True, False):
            if not complete:
                (self.root / f"llama-ffm/target/llama-ffm-{VERSION}.jar").unlink()
            output = self.root / f"github-output-{complete}"
            destination = self.root / f"collected-{complete}"
            result = subprocess.run([sys.executable, str(scripts / "collect-release-assets.py"), str(destination)],
                                    env={**os.environ, "GITHUB_OUTPUT": str(output)},
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0 if complete else 1, result.stdout + result.stderr)
            self.assertEqual(output.read_text(), f"complete={str(complete).lower()}\n")
            self.assertTrue((destination / f"llama-{VERSION}.jar").is_file())


if __name__ == "__main__":
    unittest.main()
