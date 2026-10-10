# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

import os
import subprocess
import sys
import unittest

from buildcheck import natives, workflow
from buildcheck.tests.helpers import REPO, upload, workflow_text

CSV = """# a comment
classifier,directory,library,platform,kind
cpu-linux-x86-64,Linux/x86_64/cpu,libjllama.so,yes,library
cuda13-linux-x86-64,Linux/x86_64/cuda13,libggml-cuda.so,no,module
cpu-linux-s390x,Linux/s390x/cpu,libjllama.so,yes,library
cpu-windows-aarch64,Windows/aarch64/cpu,jllama.dll,yes,library
vulkan-windows-aarch64,Windows/aarch64/vulkan,ggml-vulkan.dll,no,module
cpu-android-aarch64,Linux-Android/aarch64/cpu,libjllama.so,no,library
opencl-android-aarch64,Linux-Android/aarch64/opencl,libggml-opencl.so,no,module
metal-macos-aarch64,Mac/aarch64/metal,libjllama.dylib,yes,library
"""

ROWS = natives.rows(CSV)
TARGETS = ["linux-x86-64", "macos-aarch64", "windows-aarch64"]


def pom(rows):
    execs = "\n".join(natives.pom_execution(r) for r in rows)
    return f"""<project xmlns="http://maven.apache.org/POM/4.0.0"><profiles><profile><id>natives</id>
<build><plugins><plugin><executions>
{execs}
</executions></plugin></plugins></build></profile></profiles></project>"""


def platform(classifiers):
    deps = "".join(f"<dependency><classifier>{c}</classifier></dependency>" for c in classifiers)
    return f'<project xmlns="http://maven.apache.org/POM/4.0.0"><dependencies>{deps}</dependencies></project>'


def download(artifact):
    return ["      - uses: actions/download-artifact@v8", "        with:", f"          name: {artifact}",
            "          path: smoke/"]


def jobs(rows=ROWS, package_needs="[build]", set_targets=TARGETS, run_targets=TARGETS):
    build = [line for r in rows for line in upload("natives-" + r["classifier"])]
    sets = [line for t in set_targets for line in upload("llama-smoke-" + t, f"smoke-sets/{t}/")]
    smoke = [line for t in run_targets for line in download("llama-smoke-" + t)]
    return workflow.parse(workflow_text([("build", None, build), ("other", None, []),
                                         ("package", package_needs, sets), ("smoke", "[package]", smoke)]))


def matrix_jobs(targets, download_name="llama-smoke-${{ matrix.target }}"):
    """The workflow of jobs(), with the smoke job as a matrix over `targets` (macOS downloaded literally)."""
    rows = "\n".join(f"          - {{ target: {t}, runner: ubuntu-latest }}" for t in targets)
    body = ["    strategy:", "      matrix:", "        include:", rows, "    steps:"] + download(download_name)
    build = [line for r in ROWS for line in upload("natives-" + r["classifier"])]
    sets = [line for t in TARGETS for line in upload("llama-smoke-" + t, f"smoke-sets/{t}/")]
    return workflow.parse(workflow_text([("build", None, build), ("package", "build", sets),
                                         ("smoke", "package", body),
                                         ("smoke-macos", "package", download("llama-smoke-macos-aarch64"))]))


class ListTest(unittest.TestCase):

    def test_rows_skip_comments(self):
        self.assertEqual(ROWS[0], {"classifier": "cpu-linux-x86-64", "directory": "Linux/x86_64/cpu",
                                   "library": "libjllama.so", "platform": "yes", "kind": "library"})
        self.assertEqual(len(ROWS), 8)
        self.assertEqual([r["classifier"] for r in natives.modules(ROWS)],
                         ["cuda13-linux-x86-64", "vulkan-windows-aarch64", "opencl-android-aarch64"])
        self.assertEqual(natives.target(ROWS[1]), "linux-x86-64")

    def test_module_name_is_unique_per_classifier(self):
        self.assertEqual(natives.module_name("sycl-linux-x86-64"), "net.ladenthin.llama.natives.sycl_linux_x86_64")

    def test_smoke_targets_are_the_desktop_library_platforms(self):
        # linux-x86-64 (cpu + cuda13), windows-aarch64 (cpu + vulkan), macos (metal); s390x has no
        # runner, Android no `java -cp`.
        self.assertEqual(natives.smoke_targets(ROWS), TARGETS)
        self.assertEqual([r["classifier"] for r in natives.smoke_set(ROWS, "linux-x86-64")],
                         ["cpu-linux-x86-64", "cuda13-linux-x86-64"])
        self.assertEqual([r["classifier"] for r in natives.smoke_set(ROWS, "macos-aarch64")], ["metal-macos-aarch64"])

    def test_check_rows(self):
        self.assertEqual(natives.check_rows(ROWS), [])
        bad = natives.rows("classifier,directory,library,platform,kind\n"
                           "cuda-linux-x86-64,Linux/x86_64/cuda13,x,no,module\n"
                           "cpu-a,A/b/cpu,libjllama.so,maybe,library\n"
                           "cpu-a,A/b/cpu,libjllama.so,no,thing\n")
        failures = natives.check_rows(bad)
        self.assertEqual(len(failures), 7, failures)
        self.assertIn("twice", failures[0])
        for expected in ("classifier must start with its directory's backend 'cuda13'",
                         "must hold one ggml backend module",
                         "platform must be yes or no",
                         "kind must be one of",
                         "two jars of A/b ship a file of the same name",
                         "Linux/x86_64 has module jars but no library jar"):
            self.assertTrue(any(expected in f for f in failures), expected)

    def test_check_rows_refuses_a_module_without_a_library_and_colliding_file_names(self):
        orphan = natives.rows("classifier,directory,library,platform,kind\n"
                              "vulkan-linux-aarch64,Linux/aarch64/vulkan,libggml-vulkan.so,no,module\n")
        self.assertEqual(natives.check_rows(orphan),
                         ["natives.csv: Linux/aarch64 has module jars but no library jar -- nothing would load them"])
        twice = natives.rows("classifier,directory,library,platform,kind\n"
                             "cpu-linux-x86-64,Linux/x86_64/cpu,libjllama.so,yes,library\n"
                             "sycl-linux-x86-64,Linux/x86_64/sycl,libggml-sycl.so,no,module\n"
                             "sycl-fp32-linux-x86-64,Linux/x86_64/sycl-fp32,libggml-sycl.so,no,module\n")
        failures = natives.check_rows(twice)
        self.assertEqual(len(failures), 1, failures)
        self.assertIn("same name", failures[0])
        platform_module = natives.rows("classifier,directory,library,platform,kind\n"
                                       "cpu-linux-x86-64,Linux/x86_64/cpu,libjllama.so,yes,library\n"
                                       "cuda13-linux-x86-64,Linux/x86_64/cuda13,libggml-cuda.so,yes,module\n")
        self.assertIn("cannot be a platform jar", natives.check_rows(platform_module)[0])


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
        sets = [line for t in TARGETS for line in upload("llama-smoke-" + t, f"smoke-sets/{t}/")]
        smoke = [line for t in TARGETS for line in download("llama-smoke-" + t)]
        wf = workflow_text([("build", None, [line for r in ROWS for line in upload("natives-" + r["classifier"])]),
                            ("mid", "build", []), ("package", "mid", sets), ("smoke", "package", smoke)])
        self.assertEqual(natives.check_workflow(ROWS, workflow.parse(wf)), [])

    def test_every_smoke_target_is_uploaded_and_launched(self):
        self.assertEqual(natives.check_workflow(ROWS, jobs(set_targets=TARGETS[1:])),
                         ["publish.yml package smoke-set uploads: missing linux-x86-64"])
        self.assertEqual(natives.check_workflow(ROWS, jobs(run_targets=TARGETS + ["linux-s390x"])),
                         ["publish.yml smoke runs (a job downloading llama-smoke-<target>): linux-s390x is not in "
                          ".github/natives.csv"])
        self.assertEqual(natives.check_workflow(ROWS, jobs(run_targets=TARGETS[:2])),
                         ["publish.yml smoke runs (a job downloading llama-smoke-<target>): missing windows-aarch64"])

    def test_a_matrix_launches_its_rows(self):
        self.assertEqual(natives.check_workflow(ROWS, matrix_jobs(["linux-x86-64", "windows-aarch64"])), [])
        self.assertEqual(natives.check_workflow(ROWS, matrix_jobs(["linux-x86-64"])),
                         ["publish.yml smoke runs (a job downloading llama-smoke-<target>): missing windows-aarch64"])
        # a matrix that downloads one fixed set launches that set, not its rows
        self.assertEqual(natives.check_workflow(ROWS, matrix_jobs(["windows-aarch64"], "llama-smoke-linux-x86-64")),
                         ["publish.yml smoke runs (a job downloading llama-smoke-<target>): missing windows-aarch64"])

    def test_a_module_jar_without_a_smoke_target_is_refused(self):
        rows = natives.rows(CSV + "vulkan-linux-s390x,Linux/s390x/vulkan,libggml-vulkan.so,no,module\n")
        failures = natives.check_workflow(rows, jobs(rows=rows))
        self.assertEqual(failures, ["natives.csv: the module jar vulkan-linux-s390x has no smoke target -- it would "
                                    "ship unlaunched"])

    def test_loader(self):
        text = ('List<String> LIBRARY_BACKENDS = Arrays.asList("metal", "cpu");\n'
                'List<String> MODULE_BACKENDS = Arrays.asList(\n "cuda13", "vulkan",\n "opencl");')
        self.assertEqual(natives.check_loader(ROWS, text), [])
        self.assertEqual(natives.check_loader(ROWS, text.replace('"cuda13", ', "")),
                         ["LlamaLoader.MODULE_BACKENDS does not know 'cuda13' -- its jars would never load"])
        self.assertEqual(natives.check_loader(ROWS, text.replace('"opencl"', '"opencl", "msvc"')),
                         ["LlamaLoader.MODULE_BACKENDS names 'msvc', which no module jar of natives.csv ships"])
        self.assertEqual(natives.check_loader(ROWS, text.replace('"metal", ', "")),
                         ["LlamaLoader.LIBRARY_BACKENDS does not know 'metal' -- its jars would never load"])

    def test_cmake_both_directions(self):
        text = "\n".join(f"    set(JLLAMA_BACKEND {b})" for b in ("cuda13", "cpu", "vulkan", "opencl", "metal"))
        self.assertEqual(natives.check_cmake(ROWS, text), [])
        failures = natives.check_cmake(ROWS, text.replace("cuda13", "cuda14"))
        self.assertEqual(len(failures), 2)
        self.assertIn("never sets JLLAMA_BACKEND cuda13", failures[0])
        self.assertIn("sets JLLAMA_BACKEND cuda14, which no natives jar ships", failures[1])

    def test_dependency_allowlist_covers_every_library_directory_and_nothing_unshipped(self):
        cpu = ["Linux/x86_64/cpu", "Linux/s390x/cpu", "Windows/aarch64/cpu", "Linux-Android/aarch64/cpu",
               "Mac/aarch64/metal"]
        self.assertEqual(natives.check_dependency_allowlist(ROWS, dict.fromkeys(cpu)), [])
        # a module directory may carry a list (the Android OpenCL one does), a directory no jar ships may not
        self.assertEqual(natives.check_dependency_allowlist(ROWS, dict.fromkeys(cpu + ["Linux/x86_64/cuda13"])), [])
        failures = natives.check_dependency_allowlist(ROWS, dict.fromkeys(cpu[1:] + ["Linux/x86_64/cuda12"]))
        self.assertEqual(len(failures), 2, failures)
        self.assertIn("no allowlist for Linux/x86_64/cpu", failures[0])
        self.assertIn("Linux/x86_64/cuda12, which no natives jar", failures[1])

    def test_readme(self):
        text = " ".join(f"`{r['classifier']}`" for r in ROWS)
        self.assertEqual(natives.check_readme(ROWS, text), [])
        failures = natives.check_readme(ROWS, text.replace("`cpu-linux-s390x`", "cpu-linux-s390x"))
        self.assertEqual(failures, ["README.md does not document the natives jar `cpu-linux-s390x`"])

    README = "<artifactId>llama-platform</artifactId>\n    <version>5.2.0</version>\n</dependency>"

    def test_jbang_example(self):
        readme = self.README
        platform = [r["classifier"] for r in ROWS if r["platform"] == "yes"]
        lines = ["//DEPS net.ladenthin:llama:5.2.0"] + [f"//DEPS net.ladenthin:llama:5.2.0:{c}" for c in platform]
        script = "///usr/bin/env jbang\n" + "\n".join(lines) + "\n// a comment naming net.ladenthin:llama:5.2.0:cuda13-linux-x86-64 is not a dependency\n"
        self.assertEqual(natives.check_jbang_example(ROWS, script, readme), [])
        # a platform jar missing, a GPU jar named instead
        broken = script.replace(":cpu-linux-s390x", ":cuda13-linux-x86-64")
        self.assertEqual(natives.check_jbang_example(ROWS, broken, readme),
                         ["examples/jbang/Chat.java: missing the platform natives jar cpu-linux-s390x",
                          "examples/jbang/Chat.java: names cuda13-linux-x86-64, which is not a platform=yes row of .github/natives.csv"])
        # the classes jar missing
        failures = natives.check_jbang_example(ROWS, script.replace("//DEPS net.ladenthin:llama:5.2.0\n", ""), readme)
        self.assertEqual(failures, ["examples/jbang/Chat.java: the classes jar net.ladenthin:llama:<version> is missing"])
        # a version the README has moved past (the bump edits both by hand)
        failures = natives.check_jbang_example(ROWS, script, readme.replace("5.2.0", "5.3.0"))
        self.assertEqual(failures, ["examples/jbang/Chat.java: DEPS version(s) ['5.2.0'] must be the README install snippet's 5.3.0"])
        # another artifact, or JBang's BOM-only pom form
        failures = natives.check_jbang_example(ROWS, script + "//DEPS net.ladenthin:llama-platform:5.2.0@pom\n", readme)
        self.assertEqual(failures, ["examples/jbang/Chat.java: `net.ladenthin:llama-platform:5.2.0@pom` is not net.ladenthin:llama:<version>[:<classifier>]"])

    @staticmethod
    def server_pom(version="5.2.0", profiles=("cuda13-linux-x86-64", "vulkan-windows-aarch64", "opencl-android-aarch64"),
                   adds=None):
        adds = adds or {}
        body = ""
        for pid in profiles:
            deps = "".join(f"<dependency><groupId>net.ladenthin</groupId><artifactId>llama</artifactId>"
                           f"<classifier>{c}</classifier></dependency>" for c in adds.get(pid, [pid]))
            body += f"<profile><id>{pid}</id><dependencies>{deps}</dependencies></profile>"
        return (f'<project xmlns="http://maven.apache.org/POM/4.0.0"><properties><llama.version>{version}'
                f'</llama.version></properties><profiles><profile><id>assembly</id></profile>{body}</profiles></project>')

    def test_server_example(self):
        self.assertEqual(natives.check_server_example(ROWS, self.server_pom(), self.README), [])
        self.assertEqual(natives.check_server_example(ROWS, self.server_pom(version="5.1.0"), self.README),
                         ["examples/server/pom.xml: <llama.version> is 5.1.0, the README install snippet's is 5.2.0"])
        failures = natives.check_server_example(ROWS, self.server_pom(profiles=("cuda13-linux-x86-64",)), self.README)
        self.assertEqual(failures, ["examples/server/pom.xml GPU profiles: missing opencl-android-aarch64",
                                    "examples/server/pom.xml GPU profiles: missing vulkan-windows-aarch64"])
        wrong = self.server_pom(adds={"cuda13-linux-x86-64": ["cpu-linux-x86-64", "cuda13-linux-x86-64"]})
        failures = natives.check_server_example(ROWS, wrong, self.README)
        self.assertEqual(len(failures), 1, failures)
        self.assertIn("must add exactly the natives jar cuda13-linux-x86-64", failures[0])
        # a profile for a jar that does not exist
        failures = natives.check_server_example(ROWS, self.server_pom(profiles=("cuda13-linux-x86-64", "vulkan-windows-aarch64",
                                                                                 "opencl-android-aarch64", "cuda12-linux-x86-64")),
                                                self.README)
        self.assertEqual(failures, [])  # unknown ids that add unknown jars are not GPU profiles (an `assembly`-like profile)

    def test_agent_version(self):
        def pom(version):
            return ('<project xmlns="http://maven.apache.org/POM/4.0.0"><modelVersion>4.0.0</modelVersion>'
                    f'<parent><version>9</version></parent><version>{version}</version></project>')
        self.assertEqual(natives.check_agent_version(pom("5.2.0-SNAPSHOT"), pom("5.2.0-SNAPSHOT")), [])
        self.assertEqual(natives.check_agent_version(pom("5.2.0"), pom("5.2.0-SNAPSHOT")),
                         ["llama-atmosphere-agent/pom.xml is version 5.2.0-SNAPSHOT, the reactor 5.2.0: "
                          "set both to the same version"])


class RepositoryTest(unittest.TestCase):
    """The checks over this repository: what code-style runs, so a red here is a red there."""

    def test_the_repository_agrees_with_its_list(self):
        self.assertEqual(natives.check(REPO), [])

    def test_the_repository_derives_the_five_smoke_targets(self):
        rows = natives.rows(natives.read(REPO, ".github/natives.csv"))
        self.assertEqual(natives.smoke_targets(rows),
                         ["linux-aarch64", "linux-x86-64", "macos-aarch64", "windows-aarch64", "windows-x86-64"])

    def test_cli_prints_the_pom_executions_and_the_smoke_targets(self):
        cli = os.path.join(REPO, ".github", "check-natives.py")
        out = subprocess.run([sys.executable, cli, "smoke-targets"], capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.split(), ["linux-aarch64", "linux-x86-64", "macos-aarch64", "windows-aarch64",
                                              "windows-x86-64"])
        out = subprocess.run([sys.executable, cli, "smoke-set", "linux-aarch64"], capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.split(), ["cpu-linux-aarch64", "vulkan-linux-aarch64"])
        out = subprocess.run([sys.executable, cli, "pom"], capture_output=True, text=True, check=True)
        rows = natives.rows(natives.read(REPO, ".github/natives.csv"))
        self.assertEqual(out.stdout.count("<execution>"), len(rows))
        self.assertEqual(subprocess.run([sys.executable, cli, "nonsense"], capture_output=True).returncode, 2)
        self.assertEqual(subprocess.run([sys.executable, cli, "smoke-set", "linux-s390x"], capture_output=True).returncode, 2)


if __name__ == "__main__":
    unittest.main()
