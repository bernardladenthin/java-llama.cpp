# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

import os
import tempfile
import unittest

from buildcheck import aarpoms
from buildcheck.tests.helpers import REPO

NS = 'xmlns="http://maven.apache.org/POM/4.0.0"'


def dep(group, artifact, version=None, scope=None, optional=None, kind=None):
    parts = ["<groupId>%s</groupId>" % group, "<artifactId>%s</artifactId>" % artifact]
    for tag, value in (("version", version), ("type", kind), ("scope", scope), ("optional", optional)):
        if value is not None:
            parts.append("<%s>%s</%s>" % (tag, value, tag))
    return "<dependency>%s</dependency>" % "".join(parts)


def pom(deps, properties="", ns=NS):
    return "<project %s><properties>%s</properties><dependencies>%s</dependencies></project>" % (
        ns, properties, "".join(deps))


# The 6.0.0 llama-android-opencl pom's dependencies, as published.
PUBLISHED_600 = [
    ("com.fasterxml.jackson.core", "jackson-databind", "2.22.4", "compile", None),
    ("org.slf4j", "slf4j-api", "2.0.20", "compile", None),
    ("org.jspecify", "jspecify", "1.0.1", "compile", None),
    ("org.checkerframework", "checker-qual", "4.3.1", "compile", None),
    ("net.ladenthin", "llama-android", "6.0.0", "compile", "aar"),
]
EXPECTED_600 = {
    ("com.fasterxml.jackson.core", "jackson-databind"): "2.22.4",
    ("org.slf4j", "slf4j-api"): "2.0.20",
    ("org.jspecify", "jspecify"): "1.0.1",
}


class AarPomsTest(unittest.TestCase):

    def write(self, folder, name, text):
        path = os.path.join(folder, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def test_core_set_takes_compile_scope_only_and_resolves_properties_core_first(self):
        with tempfile.TemporaryDirectory() as d:
            root = self.write(d, "root.xml", pom([], "<a.version>1</a.version><b.version>root</b.version>"))
            core = self.write(d, "core.xml", pom([
                dep("g", "implicit", "${a.version}"),
                dep("g", "explicit", "${b.version}", scope="compile"),
                dep("g", "literal", "3.0"),
                dep("g", "provided", "1", scope="provided"),
                dep("g", "runtime", "1", scope="runtime"),
                dep("g", "test", "1", scope="test"),
                dep("g", "optional", "1", optional="true"),
            ], "<b.version>core</b.version>"))
            self.assertEqual(aarpoms.core_compile_dependencies(core, root), {
                ("g", "implicit"): "1", ("g", "explicit"): "core", ("g", "literal"): "3.0"})

    def test_core_set_fails_loud_on_an_undefined_property_or_a_missing_version(self):
        with tempfile.TemporaryDirectory() as d:
            root = self.write(d, "root.xml", pom([]))
            core = self.write(d, "core.xml", pom([dep("g", "a", "${nowhere}")]))
            with self.assertRaisesRegex(ValueError, "nowhere"):
                aarpoms.core_compile_dependencies(core, root)
            core = self.write(d, "core.xml", pom([dep("g", "a")]))
            with self.assertRaisesRegex(ValueError, "no <version>"):
                aarpoms.core_compile_dependencies(core, root)

    def test_reads_a_pom_without_the_maven_namespace(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.write(d, "a.pom", pom([dep("g", "a", "1", scope="compile")], ns=""))
            self.assertEqual(aarpoms.published_dependencies(path), [("g", "a", "1", "compile", None)])

    def test_the_600_pom_fails_on_checker_qual_and_nothing_else(self):
        self.assertEqual(aarpoms.violations(EXPECTED_600, PUBLISHED_600), [
            "org.checkerframework:checker-qual (scope compile) is not a compile-scope dependency of the core"])

    def test_the_fixed_pom_passes_and_the_cpu_aar_dependency_is_allowed(self):
        fixed = [d for d in PUBLISHED_600 if d[1] != "checker-qual"]
        self.assertEqual(aarpoms.violations(EXPECTED_600, fixed), [])
        self.assertEqual(aarpoms.violations(EXPECTED_600, fixed[:3]), [])  # the CPU AAR's own pom

    def test_missing_wrong_version_wrong_scope_duplicate_and_a_jar_typed_cpu_aar(self):
        published = [
            ("com.fasterxml.jackson.core", "jackson-databind", "2.0", "compile", None),
            ("org.slf4j", "slf4j-api", "2.0.20", "runtime", None),
            ("org.slf4j", "slf4j-api", "2.0.20", None, None),
            ("net.ladenthin", "llama-android", "6.0.0", "compile", None),
        ]
        self.assertEqual(aarpoms.violations(EXPECTED_600, published), [
            "com.fasterxml.jackson.core:jackson-databind is 2.0, the core has 2.22.4",
            "org.slf4j:slf4j-api has scope runtime, the core has compile",
            "org.slf4j:slf4j-api is listed twice",
            "net.ladenthin:llama-android must be <type>aar</type>, not jar",
            "org.jspecify:jspecify 1.0.1 is missing (a compile-scope dependency of the core)",
        ])

    def test_this_repository_core_set(self):
        expected = aarpoms.core_compile_dependencies(
            os.path.join(REPO, "llama", "pom.xml"), os.path.join(REPO, "pom.xml"))
        for key in (("com.fasterxml.jackson.core", "jackson-databind"), ("org.slf4j", "slf4j-api"),
                    ("org.jspecify", "jspecify")):
            self.assertIn(key, expected)
        # provided (major 55 classes) and the JVM-only runtime binding stay out of an Android consumer's graph
        self.assertNotIn(("org.checkerframework", "checker-qual"), expected)
        self.assertNotIn(("org.slf4j", "slf4j-simple"), expected)
        for version in expected.values():
            self.assertNotIn("${", version)

    def test_exit_codes(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "llama"))
            self.write(d, "pom.xml", pom([], "<j.version>1</j.version>"))
            self.write(os.path.join(d, "llama"), "pom.xml", pom([
                dep("g", "a", "${j.version}"), dep("g", "q", "4", scope="provided")]))
            good = self.write(d, "good.pom", pom([dep("g", "a", "1", scope="compile")]))
            bad = self.write(d, "bad.pom", pom([dep("g", "a", "1", scope="compile"),
                                                dep("g", "q", "4", scope="compile")]))
            self.assertEqual(aarpoms.main(["x", d, good]), 0)
            self.assertEqual(aarpoms.main(["x", d, good, bad]), 1)
            self.assertEqual(aarpoms.main(["x", d]), 2)


if __name__ == "__main__":
    unittest.main()
