# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Fail when a published Android AAR pom does not mirror the core's compile-scope dependencies.

The AAR's classes.jar is the core's classes jar, so an Android consumer needs exactly what a Maven
consumer of net.ladenthin:llama gets: the core's compile-scope, non-optional dependencies, at the
same versions -- no more, no less. The AAR poms are written by llama-android/build.gradle.kts, a
Gradle build of its own, so nothing ties them to llama/pom.xml but this check and that build's
parsing. The 6.0.0 poms shipped checker-qual at compile scope although the core keeps it provided
(major 55 classes, CLAUDE.md "Java 8 bytecode floor"): the versions were derived from the core pom,
the scopes came from a hand-kept list.

Usage (the CLI is .github/verify-aar-poms.py): verify-aar-poms.py <repo-root> <aar-pom>...

Each AAR pom may carry, besides the mirrored set, the CPU AAR itself (net.ladenthin:llama-android,
type aar), which the OpenCL AAR depends on. Exits non-zero when a pom lacks a mirrored dependency,
carries another one, or carries one at a different version or a scope other than compile. Only the
standard library, so it runs on any runner.
"""

import os
import re
import sys
import xml.etree.ElementTree as ET

# The one dependency an AAR pom may add to the mirrored set: the OpenCL AAR plugs into the CPU AAR.
CPU_AAR = ("net.ladenthin", "llama-android")
PROPERTY = re.compile(r"\$\{([^}]+)\}")


def _children(element, tag):
    """Direct children by local name, so a pom with or without the Maven namespace reads the same."""
    return [c for c in element if c.tag.rsplit("}", 1)[-1] == tag]


def _text(element, tag):
    found = _children(element, tag)
    return found[0].text.strip() if found and found[0].text else None


def _properties(root):
    props = _children(root, "properties")
    return {c.tag.rsplit("}", 1)[-1]: (c.text or "").strip() for c in props[0]} if props else {}


def _dependencies(root):
    deps = _children(root, "dependencies")
    return _children(deps[0], "dependency") if deps else []


def core_compile_dependencies(core_pom, root_pom):
    """{(groupId, artifactId): version} of the core's compile-scope, non-optional dependencies.

    A ${property} version is resolved the way llama-android/build.gradle.kts does it: from the core
    pom's properties, then from the parent's.
    """
    core = ET.parse(core_pom).getroot()
    props = _properties(ET.parse(root_pom).getroot())
    props.update(_properties(core))
    result = {}
    for dep in _dependencies(core):
        if (_text(dep, "scope") or "compile") != "compile" or _text(dep, "optional") == "true":
            continue
        key = (_text(dep, "groupId"), _text(dep, "artifactId"))
        version = _text(dep, "version")
        if version is None:
            raise ValueError("%s:%s has no <version> in %s" % (key + (core_pom,)))
        match = PROPERTY.fullmatch(version)
        if match:
            if match.group(1) not in props:
                raise ValueError("property <%s> of %s:%s is not defined" % ((match.group(1),) + key))
            version = props[match.group(1)]
        result[key] = version
    return result


def published_dependencies(aar_pom):
    """[(groupId, artifactId, version, scope, type)] in pom order; scope and type as written."""
    root = ET.parse(aar_pom).getroot()
    return [(_text(d, "groupId"), _text(d, "artifactId"), _text(d, "version"), _text(d, "scope"),
             _text(d, "type")) for d in _dependencies(root)]


def violations(expected, published):
    """What is wrong with one AAR pom's dependencies against the core's compile-scope set."""
    problems = []
    seen = set()
    for group, artifact, version, scope, kind in published:
        key = (group, artifact)
        name = "%s:%s" % key
        if key in seen:
            problems.append("%s is listed twice" % name)
        seen.add(key)
        if key == CPU_AAR:
            if kind != "aar":
                problems.append("%s must be <type>aar</type>, not %s" % (name, kind or "jar"))
            continue
        if key not in expected:
            problems.append("%s (scope %s) is not a compile-scope dependency of the core" % (name, scope or "compile"))
            continue
        if version != expected[key]:
            problems.append("%s is %s, the core has %s" % (name, version, expected[key]))
        if (scope or "compile") != "compile":
            problems.append("%s has scope %s, the core has compile" % (name, scope))
    for key in sorted(set(expected) - seen):
        problems.append("%s:%s %s is missing (a compile-scope dependency of the core)" % (key + (expected[key],)))
    return problems


def main(argv):
    if len(argv) < 3:
        print("usage: verify-aar-poms.py <repo-root> <aar-pom>...", file=sys.stderr)
        return 2
    repo = argv[1]
    expected = core_compile_dependencies(os.path.join(repo, "llama", "pom.xml"), os.path.join(repo, "pom.xml"))
    print("core compile-scope dependencies: %s" % ", ".join(
        "%s:%s:%s" % (k + (v,)) for k, v in sorted(expected.items())))
    failed = 0
    for pom in argv[2:]:
        problems = violations(expected, published_dependencies(pom))
        for p in problems:
            print("::error file=%s::%s" % (pom, p))
        print("%s: %s" % (pom, "%d problem(s)" % len(problems) if problems else "mirrors the core"))
        failed += bool(problems)
    return 1 if failed else 0
