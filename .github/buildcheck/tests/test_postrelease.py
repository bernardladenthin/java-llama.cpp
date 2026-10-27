# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

import hashlib
import io
import os
import struct
import subprocess
import tarfile
import tempfile
import unittest
import zipfile

from buildcheck import natives, postrelease

CSV = """# the natives jars
classifier,directory,library,platform,kind,optin
cpu-linux-x86-64,Linux/x86_64/cpu,libjllama.so,yes,library,no
cpu-android-aarch64,Linux-Android/aarch64/cpu,libjllama.so,no,library,no
cpu-android-x86-64,Linux-Android/x86_64/cpu,libjllama.so,no,library,no
vulkan-linux-x86-64,Linux/x86_64/vulkan,libggml-vulkan.so,no,module,no
"""
PATCHES = [("0001-a.patch", "1" * 64), ("0012-b.patch", "2" * 64)]
COMMIT = "c" * 40
UPSTREAM = "u" * 40


def elf(align):
    """A minimal little-endian ELF64 with one PT_LOAD segment of the given alignment."""
    header = b"\x7fELF\x02\x01" + b"\x00" * 26 + struct.pack("<Q", 64) + b"\x00" * 14 + struct.pack("<HH", 56, 1)
    header = header.ljust(64, b"\x00")
    return header + struct.pack("<II", 1, 5) + b"\x00" * 40 + struct.pack("<Q", align)


def zipped(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def build_txt(classifier, kind, **overrides):
    keys = {"llama.cpp": "b100", "llama.cpp.commit": UPSTREAM, "backend": classifier.split("-")[0], "kind": kind,
            "classifier": classifier, "jllama.version": "1.2.0", "jllama.commit": COMMIT,
            "patches": "0001,0012", "patches.sha256": postrelease.patches_sha256(PATCHES), "compiler": "GNU 14",
            "toolkit": "none", "signed": "no"}
    keys.update(overrides)
    return "# written by cmake\n" + "".join(f"{k}={v}\n" for k, v in keys.items())


def natives_jar(row, build=None, extra=None, manifest_name=None):
    prefix = f"net/ladenthin/llama/{row['directory']}/"
    name = manifest_name or natives.module_name(row["classifier"])
    # a long name wraps at 72 bytes in a real manifest; the check unfolds it
    manifest = f"Manifest-Version: 1.0\r\nAutomatic-Module-Name: {name[:40]}\r\n {name[40:]}\r\n"
    entries = {"META-INF/MANIFEST.MF": manifest, "META-INF/third-party/natives/NOTICE.md": "n",
               prefix + row["library"]: elf(16384), prefix + "jllama-build.txt": build or build_txt(row["classifier"], row["kind"]),
               prefix + "jllama-files.txt": row["library"], prefix + "jllama-sbom.json": "{}"}
    if row["kind"] == "library":
        entries[prefix + "libggml-base.so"] = elf(16384)
    entries.update(extra or {})
    return zipped(entries)


class Fixture(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.assets = os.path.join(tmp.name, "assets")
        self.root = os.path.join(tmp.name, "repo")
        os.makedirs(self.assets)
        os.makedirs(os.path.join(self.root, ".github"))
        with open(os.path.join(self.root, ".github", "natives.csv"), "w") as f:
            f.write(CSV)
        self.rows = natives.rows(CSV)

    def write(self, name, data):
        with open(os.path.join(self.assets, name), "wb") as f:
            f.write(data.encode() if isinstance(data, str) else data)

    def release_natives(self, **overrides):
        for row in self.rows:
            self.write(f"llama-1.2.0-{row['classifier']}.jar", overrides.get(row["classifier"]) or natives_jar(row))


class NativesTest(Fixture):
    def test_a_correct_set_passes_and_the_manifest_name_is_unfolded(self):
        self.release_natives()
        result, builds = postrelease.check_natives(self.assets, self.rows, "1.2.0")
        self.assertEqual(result.problems, [])
        self.assertEqual(sorted(builds), sorted(r["classifier"] for r in self.rows))

    def test_each_property_of_a_natives_jar(self):
        linux, vulkan = self.rows[0], self.rows[3]
        self.release_natives(**{
            "cpu-linux-x86-64": natives_jar(linux, build=build_txt("cpu-linux-x86-64", "module", compiler=""),
                                            extra={"net/ladenthin/llama/Mac/aarch64/metal/x.dylib": "x"},
                                            manifest_name="llama"),
            "vulkan-linux-x86-64": natives_jar(vulkan, extra={"net/ladenthin/llama/Linux/x86_64/vulkan/libggml.so": "x"}),
        })
        os.remove(os.path.join(self.assets, "llama-1.2.0-cpu-android-x86-64.jar"))
        problems = postrelease.check_natives(self.assets, self.rows, "1.2.0")[0].problems
        self.assertIn("cpu-linux-x86-64: files outside net/ladenthin/llama/Linux/x86_64/cpu/: "
                      "['net/ladenthin/llama/Mac/aarch64/metal/x.dylib']", problems)
        self.assertIn("cpu-linux-x86-64: jllama-build.txt lacks ['compiler']", problems)
        self.assertIn("cpu-linux-x86-64: jllama-build.txt says kind=module, expected library", problems)
        self.assertIn("cpu-linux-x86-64: Automatic-Module-Name llama, expected "
                      "net.ladenthin.llama.natives.cpu_linux_x86_64", problems)
        self.assertIn("cpu-android-x86-64: llama-1.2.0-cpu-android-x86-64.jar is not in the release", problems)
        self.assertTrue(any(p.startswith("vulkan-linux-x86-64: a module jar holds libggml-vulkan.so and the lists only")
                            for p in problems), problems)


class ProvenanceTest(unittest.TestCase):
    def builds(self, **overrides):
        out = {}
        for classifier in ("cpu-linux-x86-64", "vulkan-linux-x86-64"):
            out[classifier] = postrelease.build_info(build_txt(classifier, "library", **overrides.get(classifier, {})))
        return out

    def test_the_recomputed_patch_set_matches(self):
        result = postrelease.check_provenance(self.builds(), "1.2.0", COMMIT, PATCHES)
        self.assertEqual(result.problems, [])
        self.assertIn("signed=no", result.notes[-1])

    def test_dirty_commit_another_upstream_and_a_patch_set_that_is_not_the_tags(self):
        builds = self.builds(**{"vulkan-linux-x86-64": {"jllama.commit": COMMIT + "-dirty", "llama.cpp.commit": "x" * 40}})
        problems = postrelease.check_provenance(builds, "1.2.0", COMMIT, PATCHES + [("0013-c.patch", "3" * 64)]).problems
        self.assertIn(f"llama.cpp.commit differs between the natives jars: {sorted([UPSTREAM, 'x' * 40])}", problems)
        self.assertIn(f"vulkan-linux-x86-64: jllama.commit {COMMIT}-dirty, the tag is {COMMIT}", problems)
        self.assertIn("cpu-linux-x86-64: patches=0001,0012, the tag has 0001,0012,0013", problems)
        self.assertTrue(any("patches.sha256" in p and "recomputed from the tag" in p for p in problems))

    def test_the_hash_is_the_cmake_one(self):
        # jllama-build-info.cmake: string(SHA256) over "<name> <sha256>\n" per patch, in name order
        manifest = "0001-a.patch " + "1" * 64 + "\n0012-b.patch " + "2" * 64 + "\n"
        self.assertEqual(postrelease.patches_sha256(PATCHES), hashlib.sha256(manifest.encode()).hexdigest())
        self.assertEqual(postrelease.patch_ids(PATCHES), "0001,0012")

    def test_the_patch_set_is_read_from_the_blobs_at_the_revision(self):
        with tempfile.TemporaryDirectory() as repo:
            def git(*args):
                subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True)
            git("init", "-q")
            os.makedirs(os.path.join(repo, "llama", "patches"))
            for name, body in (("0002-b.patch", b"b\n"), ("0001-a.patch", b"a\n"), ("README.md", b"x")):
                with open(os.path.join(repo, "llama", "patches", name), "wb") as f:
                    f.write(body)
            git("add", ".")
            git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x")
            with open(os.path.join(repo, "llama", "patches", "0001-a.patch"), "wb") as f:
                f.write(b"changed in the working tree\n")
            self.assertEqual(postrelease.patch_set(repo), [("0001-a.patch", hashlib.sha256(b"a\n").hexdigest()),
                                                           ("0002-b.patch", hashlib.sha256(b"b\n").hexdigest())])


PATCH_A = """--- a/src/x.cpp
+++ b/src/x.cpp
@@ -1,1 +1,2 @@
 int x;
+int jllama_patch_a_marker = 1;
"""
PATCH_B = """--- a/src/x.cpp
+++ b/src/x.cpp
@@ -1,2 +1,2 @@
-int jllama_patch_a_marker = 1;
+int jllama_patch_b_rewrote_it = 2;
--- /dev/null
+++ b/src/new.h
@@ -0,0 +1 @@
+#define JLLAMA_NEW_HEADER 1
"""


class SourceTest(Fixture):
    def archive(self, stamp, files, owner=(0, 0, "", ""), mtimes=(100,)):
        top = "llama.cpp-b100-jllama-1.2.0-src"
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            entries = dict(files, **{".jllama-patches-applied": stamp})
            for i, (name, text) in enumerate(entries.items()):
                info = tarfile.TarInfo(f"{top}/{name}")
                data = text.encode()
                info.size, info.mtime = len(data), mtimes[i % len(mtimes)]
                info.uid, info.gid, info.uname, info.gname = owner
                tar.addfile(info, io.BytesIO(data))
        self.write(top + ".tar.gz", buffer.getvalue())

    def check(self, patches=PATCHES):
        builds = {"cpu-linux-x86-64": {"llama.cpp.commit": UPSTREAM}}
        texts = {"0001-a.patch": PATCH_A, "0012-b.patch": PATCH_B}
        return postrelease.check_source(self.assets, "1.2.0", builds, patches, texts.get)

    def stamp(self, head=UPSTREAM, patches=PATCHES):
        return f"head {head}\n" + "".join(f"{n} {h}\n" for n, h in patches) + "tree abc\n"

    def test_a_reproducible_archive_with_every_patch_in_it_passes(self):
        # 0012 rewrote 0001's line: x.cpp is held to 0012, 0001 to "one added line survives" -- which fails
        # here, as it must when a later patch removed everything an earlier one added
        self.archive(self.stamp(), {"src/x.cpp": "int x;\nint jllama_patch_b_rewrote_it = 2;\n",
                                    "src/new.h": "#define JLLAMA_NEW_HEADER 1\n"})
        self.assertEqual(self.check().problems,
                         ["0001-a.patch: none of its added lines survives in ['src/x.cpp']"])
        self.archive(self.stamp(), {"src/x.cpp": "int x;\nint jllama_patch_a_marker = 1;\nint jllama_patch_b_rewrote_it = 2;\n",
                                    "src/new.h": "#define JLLAMA_NEW_HEADER 1\n"})
        self.assertEqual(self.check().problems, [])

    def test_stamp_head_stamp_patches_missing_patch_content_and_a_non_reproducible_pack(self):
        self.archive(self.stamp(head="z" * 40, patches=PATCHES[:1]), {"src/x.cpp": "int x;\n"},
                     owner=(1000, 1000, "runner", "docker"), mtimes=(100, 200))
        problems = self.check().problems
        self.assertIn("2 different mtimes (a reproducible pack has one)", problems)
        self.assertTrue(any(p.startswith("owners [(1000, 1000, 'runner', 'docker')]") for p in problems))
        self.assertIn(f"the stamp's head {'z' * 40}, the natives jars say ['{UPSTREAM}']", problems)
        self.assertTrue(any(p.startswith("the stamp's patches differ from the tag's") for p in problems))
        self.assertIn("0012-b.patch: src/new.h is not in the archive", problems)
        self.assertIn("0012-b.patch: none of its added lines is in src/x.cpp", problems)


class AndroidTest(Fixture):
    def setUp(self):
        super().setUp()
        with open(os.path.join(self.root, "pom.xml"), "w") as f:
            f.write("<project><properties><j.version>1</j.version></properties></project>")
        os.makedirs(os.path.join(self.root, "llama"))
        with open(os.path.join(self.root, "llama", "pom.xml"), "w") as f:
            f.write("<project><dependencies><dependency><groupId>g</groupId><artifactId>j</artifactId>"
                    "<version>${j.version}</version></dependency></dependencies></project>")

    def release(self, cpu_jni_align=16384, opencl_classes=None, extra_dep="", opencl_cpu_version="1.2.0"):
        self.release_natives()
        core = {"net/ladenthin/llama/A.class": b"A", "module-info.class": b"M"}
        self.write("llama-1.2.0.jar", zipped(core))
        manifest = '<manifest><uses-sdk android:minSdkVersion="28" /></manifest>'
        jni = {}
        for abi, classifier in postrelease.ANDROID_ABIS.items():
            row = next(r for r in self.rows if r["classifier"] == classifier)
            with zipfile.ZipFile(os.path.join(self.assets, f"llama-1.2.0-{classifier}.jar")) as jar:
                prefix = f"net/ladenthin/llama/{row['directory']}/"
                for n in jar.namelist():
                    if n.startswith(prefix) and n.endswith(".so"):
                        jni[f"jni/{abi}/{n[len(prefix):]}"] = jar.read(n)
        jni["jni/x86_64/libggml-base.so"] = elf(cpu_jni_align)
        self.write("llama-android-1.2.0.aar", zipped({"AndroidManifest.xml": manifest, "proguard.txt": "-keep x",
                                                      "classes.jar": zipped({"net/ladenthin/llama/A.class": b"A"}), **jni}))
        self.write("llama-android-opencl-1.2.0.aar", zipped({
            "AndroidManifest.xml": manifest, "proguard.txt": "-keep x",
            "classes.jar": zipped(opencl_classes or {"META-INF/MANIFEST.MF": "x"}),
            "jni/arm64-v8a/libggml-opencl.so": elf(16384)}))
        dep = "<dependency><groupId>g</groupId><artifactId>j</artifactId><version>1</version><scope>compile</scope></dependency>"
        self.write("llama-android-1.2.0.pom", f"<project><dependencies>{dep}{extra_dep}</dependencies></project>")
        self.write("llama-android-opencl-1.2.0.pom", f"<project><dependencies>{dep}{extra_dep}<dependency><groupId>net.ladenthin"
                   f"</groupId><artifactId>llama-android</artifactId><version>{opencl_cpu_version}</version><type>aar</type>"
                   "<scope>compile</scope></dependency></dependencies></project>")

    def test_a_correct_pair_passes(self):
        self.release()
        result = postrelease.check_android(self.assets, "1.2.0", self.root)
        self.assertEqual(result.problems, [])

    def test_alignment_jni_bytes_opencl_classes_and_the_poms(self):
        checker = ("<dependency><groupId>org.checkerframework</groupId><artifactId>checker-qual</artifactId>"
                   "<version>4</version><scope>compile</scope></dependency>")
        self.release(cpu_jni_align=4096, opencl_classes={"a/A.class": b"A"}, extra_dep=checker, opencl_cpu_version="1.1.0")
        problems = postrelease.check_android(self.assets, "1.2.0", self.root).problems
        self.assertIn("llama-android.aar!jni/x86_64/libggml-base.so: LOAD alignment [4096] (Google Play needs 16 KB)", problems)
        self.assertTrue(any(p.startswith("llama-android jni/x86_64/ differs from cpu-android-x86-64") for p in problems))
        self.assertIn("llama-android-opencl: classes.jar is not empty (duplicating the API breaks D8)", problems)
        self.assertIn("llama-android pom: org.checkerframework:checker-qual (scope compile) is not a compile-scope "
                      "dependency of the core", problems)
        self.assertTrue(any("the CPU AAR dependency is" in p and "expected version 1.2.0" in p for p in problems))


class CacheTest(unittest.TestCase):
    def test_off_everywhere_passes_and_a_job_with_the_cache_on_fails(self):
        logs = {"Build Linux": "  USE_CACHE: false\nbuild.sh: sccache OFF, building with -j4\n", "Tests": "ok\n"}
        self.assertEqual(postrelease.check_cache(logs).problems, [])
        logs["Build Windows"] = "  USE_CACHE: true\nbuild.bat: sccache probe OK (wrapped cl.exe).\n"
        self.assertEqual(postrelease.check_cache(logs).problems,
                         ["Build Windows: USE_CACHE: true, build.bat: sccache probe OK"])
        self.assertEqual(postrelease.check_cache({"Tests": "ok"}).problems,
                         ["no job log shows the cache off -- the evidence is missing, not the cache"])


class JobLogTest(unittest.TestCase):
    def run_with(self, results):
        calls = []

        def fake(cmd, capture_output):
            calls.append(cmd)
            code, out, err = results[len(calls) - 1]
            return subprocess.CompletedProcess(cmd, code, out, err)
        original = postrelease.subprocess.run
        postrelease.subprocess.run = fake
        try:
            return postrelease.job_log("o/r", 7, attempts=3, pause=0), calls
        finally:
            postrelease.subprocess.run = original

    def test_retries_a_failed_read(self):
        log, calls = self.run_with([(1, b"", b"HTTP 502"), (0, b"the log", b"")])
        self.assertEqual(log, b"the log")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], ["gh", "api", "repos/o/r/actions/jobs/7/logs"])

    def test_passes_the_escape_flag(self):
        calls = []

        def fake(cmd, capture_output, text=False):
            calls.append(cmd)
            if cmd[-1] == "--help":
                return subprocess.CompletedProcess(cmd, 0, "  --allow-escape-sequences   Allow ...\n", "")
            return subprocess.CompletedProcess(cmd, 0, b"log", b"")
        original = postrelease.subprocess.run
        postrelease.subprocess.run = fake
        try:
            flags = postrelease.escape_flag()
            postrelease.job_log("o/r", 7, flags, pause=0)
        finally:
            postrelease.subprocess.run = original
        self.assertEqual(flags, ["--allow-escape-sequences"])
        self.assertEqual(calls[-1], ["gh", "api", "--allow-escape-sequences", "repos/o/r/actions/jobs/7/logs"])

    def test_names_what_gh_said(self):
        with self.assertRaisesRegex(RuntimeError, r"job 7 could not be read \(3 attempts\): HTTP 403: Forbidden$"):
            self.run_with([(1, b"", b"HTTP 403: Forbidden\n")] * 3)


if __name__ == "__main__":
    unittest.main()
