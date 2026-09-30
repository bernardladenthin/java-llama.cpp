# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT

import os
import subprocess
import sys
import unittest

from buildcheck import natives, workflow
from buildcheck.tests.helpers import REPO, upload, workflow_text

CSV = """# a comment
classifier,directory,library,platform
cpu-linux-x86-64,Linux/x86_64/cpu,libjllama.so,yes
cuda13-linux-x86-64,Linux/x86_64/cuda13,libjllama.so,no
cpu-linux-s390x,Linux/s390x/cpu,libjllama.so,yes
cpu-windows-x86,Windows/x86/cpu,jllama.dll,yes
msvc-windows-x86,Windows/x86/msvc,jllama.dll,no
cpu-android-aarch64,Linux-Android/aarch64/cpu,libjllama.so,no
opencl-android-aarch64,Linux-Android/aarch64/opencl,libjllama.so,no
metal-macos-aarch64,Mac/aarch64/metal,libjllama.dylib,yes
"""

ROWS = natives.rows(CSV)


def pom(rows):
    execs = "\n".join(natives.pom_execution(r) for r in rows)
    return f"""<project xmlns="http://maven.apache.org/POM/4.0.0"><profiles><profile><id>natives</id>
<build><plugins><plugin><executions>
{execs}
</executions></plugin></plugins></build></profile></profiles></project>"""


def platform(classifiers):
    deps = "".join(f"<dependency><classifier>{c}</classifier></dependency>" for c in classifiers)
    return f'<project xmlns="http://maven.apache.org/POM/4.0.0"><dependencies>{deps}</dependencies></project>'


def jobs(rows=ROWS, package_needs="[build]", smoke_targets=("linux-x86-64",), run_targets=("linux-x86-64",)):
    build = [line for r in rows for line in upload("natives-" + r["classifier"])]
    fatjars = [line for t in smoke_targets for line in upload("s-" + t, f"f/llama-*-all-{t}-jar-with-dependencies.jar")]
    smoke = [f"      - run: .github/smoke-test-fatjar.sh f 'llama-*-all-{t}-jar-with-dependencies.jar'"
             for t in run_targets]
    return workflow.parse(workflow_text([("build", None, build), ("other", None, []),
                                         ("package", package_needs, []),
                                         ("package-fatjars", "[package]", fatjars),
                                         ("smoke", "[package-fatjars]", smoke)]))


class ListTest(unittest.TestCase):

    def test_rows_skip_comments(self):
        self.assertEqual(ROWS[0], {"classifier": "cpu-linux-x86-64", "directory": "Linux/x86_64/cpu",
                                   "library": "libjllama.so", "platform": "yes"})
        self.assertEqual(len(ROWS), 8)

    def test_module_name_is_unique_per_classifier(self):
        self.assertEqual(natives.module_name("sycl-fp16-linux-x86-64"), "net.ladenthin.llama.natives.sycl_fp16_linux_x86_64")

    def test_fatjar_targets_need_a_second_backend_and_skip_msvc_and_android(self):
        # linux-x86-64: cpu + cuda13. windows-x86: cpu + msvc (excluded). android: excluded.
        # s390x and macos: one backend each.
        self.assertEqual(natives.fatjar_targets(ROWS), ["linux-x86-64"])

    def test_fatjar_names(self):
        text = "llama-*-all-linux-x86-64-jar-with-dependencies.jar llama-<v>-all-<os>-<arch>-jar-with-dependencies.jar " \
               "llama-${llama.version}-all-windows-aarch64-jar-with-dependencies.jar llama-5.2.0-jar-with-dependencies.jar"
        self.assertEqual(natives.fatjar_names(text), {"linux-x86-64", "windows-aarch64"})

    def test_check_rows(self):
        self.assertEqual(natives.check_rows(ROWS), [])
        bad = natives.rows("classifier,directory,library,platform\n"
                           "cuda-linux-x86-64,Linux/x86_64/cuda13,x,no\ncpu-a,A/b/cpu,x,maybe\ncpu-a,A/b/cpu,x,no\n")
        failures = natives.check_rows(bad)
        self.assertEqual(len(failures), 3, failures)
        self.assertIn("twice", failures[0])


class ConsumerTest(unittest.TestCase):

    def test_pom(self):
        self.assertEqual(natives.check_pom(ROWS, pom(ROWS)), [])
        self.assertIn("missing cuda13-linux-x86-64", natives.check_pom(ROWS, pom(ROWS[:1] + ROWS[2:]))[0])
        wrong = pom(ROWS).replace("Linux/x86_64/cuda13/**", "Linux/x86_64/cuda/**")
        self.assertIn("cuda13-linux-x86-64 has", natives.check_pom(ROWS, wrong)[0])

    def test_platform(self):
        yes = [r["classifier"] for r in ROWS if r["platform"] == "yes"]
        self.assertEqual(natives.check_platform(ROWS, platform(yes)), [])
        self.assertEqual(natives.check_platform(ROWS, platform(yes + ["cuda13-linux-x86-64"])),
                         ["llama-platform/pom.xml: cuda13-linux-x86-64 is not in .github/natives.csv"])

    def test_workflow_passes_when_everything_is_wired(self):
        self.assertEqual(natives.check_workflow(ROWS, jobs()), [])

    def test_workflow_needs_an_upload_per_row(self):
        failures = natives.check_workflow(ROWS, jobs(rows=ROWS[1:]))
        self.assertIn("publish.yml natives-* uploads: missing cpu-linux-x86-64", failures)

    def test_package_must_wait_for_every_natives_build(self):
        failures = natives.check_workflow(ROWS, jobs(package_needs="[other]"))
        self.assertTrue(failures)
        self.assertTrue(all("package does not wait for build" in f for f in failures), failures)

    def test_package_may_wait_transitively(self):
        wf = workflow_text([("build", None, [line for r in ROWS for line in upload("natives-" + r["classifier"])]),
                            ("mid", "build", []), ("package", "mid", []),
                            ("package-fatjars", "package",
                             upload("s", "f/llama-*-all-linux-x86-64-jar-with-dependencies.jar")),
                            ("smoke", "package-fatjars",
                             ["      - run: .github/smoke-test-fatjar.sh f 'llama-*-all-linux-x86-64-jar-with-dependencies.jar'"])])
        self.assertEqual(natives.check_workflow(ROWS, workflow.parse(wf)), [])

    def test_every_fatjar_target_is_uploaded_and_launched(self):
        self.assertEqual(natives.check_workflow(ROWS, jobs(smoke_targets=())),
                         ["publish.yml package-fatjars smoke-jar uploads: missing linux-x86-64"])
        self.assertEqual(natives.check_workflow(ROWS, jobs(run_targets=("linux-x86-64", "linux-s390x"))),
                         ["publish.yml fat-jar smoke runs: linux-s390x is not in .github/natives.csv"])

    def test_loader(self):
        text = 'List<String> BACKEND_PRIORITY = Arrays.asList(\n "cuda13", "msvc",\n "metal", "opencl", "cpu");'
        self.assertEqual(natives.check_loader(ROWS, text), [])
        self.assertEqual(natives.check_loader(ROWS, text.replace('"cuda13", ', "")),
                         ["LlamaLoader.BACKEND_PRIORITY does not try 'cuda13' -- its jars would never load"])

    def test_cmake_both_directions(self):
        text = "\n".join(f"    set(JLLAMA_BACKEND {b})" for b in ("cuda13", "cpu", "msvc", "opencl", "metal"))
        self.assertEqual(natives.check_cmake(ROWS, text), [])
        failures = natives.check_cmake(ROWS, text.replace("cuda13", "cuda14"))
        self.assertEqual(len(failures), 2)
        self.assertIn("never sets JLLAMA_BACKEND cuda13", failures[0])
        self.assertIn("sets JLLAMA_BACKEND cuda14, which no natives jar ships", failures[1])

    def test_dependency_allowlist_covers_every_cpu_directory_and_nothing_unshipped(self):
        cpu = ["Linux/x86_64/cpu", "Linux/s390x/cpu", "Windows/x86/cpu", "Windows/x86/msvc",
               "Linux-Android/aarch64/cpu", "Mac/aarch64/metal"]
        self.assertEqual(natives.check_dependency_allowlist(ROWS, dict.fromkeys(cpu)), [])
        # a GPU directory may carry a list (the Android OpenCL build does), a directory no jar ships may not
        self.assertEqual(natives.check_dependency_allowlist(ROWS, dict.fromkeys(cpu + ["Linux/x86_64/cuda13"])), [])
        failures = natives.check_dependency_allowlist(ROWS, dict.fromkeys(cpu[1:] + ["Linux/x86_64/cuda12"]))
        self.assertEqual(len(failures), 2, failures)
        self.assertIn("no allowlist for Linux/x86_64/cpu", failures[0])
        self.assertIn("Linux/x86_64/cuda12, which no natives jar", failures[1])

    def test_readme(self):
        text = " ".join(f"`{r['classifier']}`" for r in ROWS) + " llama-<v>-all-linux-x86-64-jar-with-dependencies.jar"
        self.assertEqual(natives.check_readme(ROWS, text), [])
        failures = natives.check_readme(ROWS, text.replace("`cpu-linux-s390x`", "cpu-linux-s390x"))
        self.assertEqual(failures, ["README.md does not document the natives jar `cpu-linux-s390x`"])

    def test_agent_class_path(self):
        good = "<Class-Path>llama-${v}-all-linux-x86-64-jar-with-dependencies.jar llama-${v}-jar-with-dependencies.jar</Class-Path>"
        self.assertEqual(natives.check_agent_class_path(ROWS, good), [])
        self.assertEqual(natives.check_agent_class_path(ROWS, "<Class-Path></Class-Path>"),
                         ["llama-atmosphere-agent/pom.xml Class-Path: missing linux-x86-64"])


class RepositoryTest(unittest.TestCase):
    """The checks over this repository: what code-style runs, so a red here is a red there."""

    def test_the_repository_agrees_with_its_list(self):
        self.assertEqual(natives.check(REPO), [])

    def test_the_repository_derives_the_four_release_targets(self):
        rows = natives.rows(natives.read(REPO, ".github/natives.csv"))
        self.assertEqual(natives.fatjar_targets(rows),
                         ["linux-aarch64", "linux-x86-64", "windows-aarch64", "windows-x86-64"])

    def test_cli_prints_the_pom_executions_and_the_targets(self):
        cli = os.path.join(REPO, ".github", "check-natives.py")
        out = subprocess.run([sys.executable, cli, "fatjar-targets"], capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.split(), ["linux-aarch64", "linux-x86-64", "windows-aarch64", "windows-x86-64"])
        out = subprocess.run([sys.executable, cli, "pom"], capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.count("<execution>"), 26)
        self.assertEqual(subprocess.run([sys.executable, cli, "nonsense"], capture_output=True).returncode, 2)


if __name__ == "__main__":
    unittest.main()
