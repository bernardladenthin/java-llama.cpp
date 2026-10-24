# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

import base64
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from buildcheck import workflow
from buildcheck.tests.helpers import REPO


GIT = shutil.which("git")
BASH = shutil.which("bash")
TOKEN = "mirror-dummy-token-for-tests"
URL = "https://github.com/owner/mirror.git"


@unittest.skipUnless(GIT and BASH, "git and bash are required")
class MirrorTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.log = self.root / "calls.jsonl"
        self.env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
                    "GIT_CONFIG_COUNT": "0", "GIT_TERMINAL_PROMPT": "0", "GITHUB_ACTIONS": "false",
                    "LLAMA_MIRROR_TOKEN": TOKEN}
        self.git("init", "-q", "--object-format=sha1")
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                 "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-qm", "fixture")
        self.sha = self.git("rev-parse", "HEAD").strip()
        (self.root / "llama").mkdir()
        (self.root / "llama/CMakeLists.txt").write_text(
            f'set(JLLAMA_GIT_TAG b1)\nset(JLLAMA_GIT_COMMIT "{self.sha}")\n')
        scripts = self.root / ".github"
        scripts.mkdir()
        self.script = scripts / "mirror-llama-source.sh"
        shutil.copyfile(Path(REPO) / ".github/mirror-llama-source.sh", self.script)

        # Like checkout v7: credentials in an included file, with a URL-specific header.
        credentials = self.root / "checkout-credentials.config"
        subprocess.run([GIT, "config", "--file", str(credentials),
                        "http.https://github.com/.extraheader", "AUTHORIZATION: basic CHECKOUT_DUMMY"],
                       check=True, env=self.env)
        self.git("config", "include.path", str(credentials))
        # A stale exact-URL setting must also be replaced, not appended as a second auth header.
        self.git("config", f"http.{URL}.extraheader", "AUTHORIZATION: basic STALE_DUMMY")
        self.git("config", f"credential.{URL}.helper", "unused-dummy-helper")
        self.config_before = (self.source / ".git/config").read_bytes()
        self.credentials_before = credentials.read_bytes()

        # Real Git handles every local operation and the config lookup. Network operations are
        # intercepted before Git can run them, so this suite cannot contact or write to GitHub.
        fake = self.root / "bin/git"
        fake.parent.mkdir()
        fake.write_text('''#!/usr/bin/env python3
import json, os, subprocess, sys
args = sys.argv[1:]
operation = next((value for value in ("ls-remote", "push") if value in args), None)
if operation is None:
    raise SystemExit(subprocess.call([os.environ["REAL_GIT"], *args]))
prefix = args[:args.index(operation)]
url = next(value for value in args if value.startswith("https://"))
def config(key):
    result = subprocess.run([os.environ["REAL_GIT"], *prefix, "config", "--get-urlmatch", key, url],
                            text=True, capture_output=True)
    return [line for line in result.stdout.splitlines() if line]
with open(os.environ["MIRROR_TEST_LOG"], "a") as log:
    log.write(json.dumps({"operation": operation, "args": args,
                          "headers": config("http.extraheader"),
                          "helpers": config("credential.helper"),
                          "prompt": os.environ.get("GIT_TERMINAL_PROMPT")}) + "\\n")
if os.environ.get("MIRROR_TEST_FAIL") == operation:
    print("simulated authentication/transport failure", file=sys.stderr)
    raise SystemExit(128)
if operation == "ls-remote" and os.environ.get("MIRROR_TEST_EXISTING"):
    print(os.environ["MIRROR_TEST_EXISTING"] + "\\trefs/tags/jllama/b1")
''')
        fake.chmod(0o755)
        self.env.update({"REAL_GIT": GIT, "MIRROR_TEST_LOG": str(self.log),
                         "PATH": str(fake.parent) + os.pathsep + os.environ["PATH"]})

    def git(self, *args):
        result = subprocess.run([GIT, "-C", str(self.source), *args], env=self.env,
                                capture_output=True, text=True, check=True)
        return result.stdout

    def run_script(self, **environment):
        return subprocess.run([BASH, str(self.script), str(self.source), "owner/mirror"],
                              env={**self.env, **environment}, capture_output=True, text=True)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_explicit_mirror_token_replaces_inherited_credentials_without_persisting_or_argv_secret(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.calls()
        self.assertEqual([call["operation"] for call in calls], ["ls-remote", "push"])
        encoded = base64.b64encode(("x-access-token:" + TOKEN).encode()).decode()
        for call in calls:
            self.assertEqual(call["headers"], ["AUTHORIZATION: basic " + encoded])
            self.assertEqual(call["helpers"], [])
            self.assertEqual(call["prompt"], "0")
            self.assertNotIn(TOKEN, " ".join(call["args"]))
            self.assertNotIn(encoded, " ".join(call["args"]))
        self.assertNotIn(TOKEN, result.stdout + result.stderr)
        self.assertNotIn(encoded, result.stdout + result.stderr)
        self.assertEqual((self.source / ".git/config").read_bytes(), self.config_before)
        self.assertEqual((self.root / "checkout-credentials.config").read_bytes(), self.credentials_before)
        self.assertEqual(calls[-1]["args"][-1], self.sha + ":refs/tags/jllama/b1")
        self.assertNotIn("--force", calls[-1]["args"])

    def test_existing_matching_tag_needs_no_push(self):
        result = self.run_script(MIRROR_TEST_EXISTING=self.sha)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([call["operation"] for call in self.calls()], ["ls-remote"])

    def test_existing_different_tag_is_never_overwritten(self):
        result = self.run_script(MIRROR_TEST_EXISTING="a" * 40)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("one number, one commit", result.stderr)
        self.assertEqual([call["operation"] for call in self.calls()], ["ls-remote"])

    def test_failed_remote_read_is_not_treated_as_an_absent_tag(self):
        result = self.run_script(MIRROR_TEST_FAIL="ls-remote")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([call["operation"] for call in self.calls()], ["ls-remote"])

    def test_failed_push_remains_a_failure(self):
        result = self.run_script(MIRROR_TEST_FAIL="push")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("mirror-llama-source: pushed", result.stdout)

    def test_missing_token_and_wrong_source_are_refused_before_any_network_call(self):
        result = self.run_script(LLAMA_MIRROR_TOKEN="")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("LLAMA_MIRROR_TOKEN is empty", result.stderr)
        (self.root / "llama/CMakeLists.txt").write_text(
            'set(JLLAMA_GIT_TAG b1)\nset(JLLAMA_GIT_COMMIT "' + "a" * 40 + '")\n')
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls(), [])

    def test_both_mirror_checkouts_disable_persisted_credentials(self):
        jobs = workflow.parse((Path(REPO) / ".github/workflows/publish.yml").read_text())
        checkouts = [step for step in jobs["mirror-llama-source"].steps()
                     if any("uses: actions/checkout@" in line for line in step)]
        self.assertEqual(len(checkouts), 2)
        for step in checkouts:
            self.assertIn("          persist-credentials: false", step)


if __name__ == "__main__":
    unittest.main()
