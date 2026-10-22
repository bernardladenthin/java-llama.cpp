# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

import os
import shutil
import subprocess
import tempfile
import unittest
import zipfile

from buildcheck.tests.helpers import REPO

BASH = shutil.which("bash")
UNZIP = shutil.which("unzip")
SMOKE_LIB = os.path.join(REPO, ".github", "smoke-lib.sh")

# Two rows of natives.csv: the Linux x86-64 library jar and one of its GPU module jars.
LIBRARY = ("cpu-linux-x86-64", "Linux/x86_64/cpu", "libjllama.so")
MODULE = ("vulkan-linux-x86-64", "Linux/x86_64/vulkan", "libggml-vulkan.so")


def write_jar(path, directory, files):
    with zipfile.ZipFile(path, "w") as jar:
        for name, text in files.items():
            jar.writestr("net/ladenthin/llama/%s/%s" % (directory, name), text)


@unittest.skipUnless(BASH and UNZIP, "needs bash and unzip")
class NativesDirectoryTest(unittest.TestCase):
    """smoke_natives_directory: the natives jars of a smoke set unpacked into one directory."""

    def setUp(self):
        self.work = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.work)
        smoke = os.path.join(self.work, "smoke")
        os.mkdir(smoke)
        write_jar(os.path.join(smoke, "llama-1.0-%s.jar" % LIBRARY[0]), LIBRARY[1],
                  {LIBRARY[2]: "library", "jllama-files.txt": "libggml.so\n"})
        write_jar(os.path.join(smoke, "llama-1.0-%s.jar" % MODULE[0]), MODULE[1],
                  {MODULE[2]: "module", "jllama-files.txt": MODULE[2] + "\n",
                   "jllama-build.txt": "llama.cpp=b1\n"})

    def unpack(self, set_dir):
        out = os.path.join(self.work, "out")
        script = 'source "$1" && smoke_natives_directory "$2" "$3"'
        return subprocess.run([BASH, "-c", script, "bash", SMOKE_LIB, set_dir, out],
                              cwd=self.work, capture_output=True, text=True), out

    def assert_unpacked(self, result, out):
        self.assertEqual(result.returncode, 0, result.stderr)
        lib_dir = os.path.join(out, "library", "net", "ladenthin", "llama", LIBRARY[1])
        self.assertEqual(result.stdout.strip(), lib_dir)
        self.assertTrue(os.path.isfile(os.path.join(lib_dir, LIBRARY[2])))
        self.assertTrue(os.path.isfile(os.path.join(lib_dir, MODULE[2])))
        self.assertFalse(os.path.exists(os.path.join(lib_dir, "jllama-build.txt")))
        with open(os.path.join(lib_dir, "jllama-files.txt")) as files:
            self.assertEqual(files.read().split(), ["libggml.so", MODULE[2]])

    def test_relative_set_directory(self):
        # publish.yml passes the set as `smoke`, relative to the job's directory; the jars are
        # unpacked after a cd into the output directory (red in run 39799320257).
        self.assert_unpacked(*self.unpack("smoke"))

    def test_absolute_set_directory(self):
        self.assert_unpacked(*self.unpack(os.path.join(self.work, "smoke")))

    def test_missing_set_directory_fails(self):
        result, _ = self.unpack("no-such-set")
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
