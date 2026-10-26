#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# Loads one Linux library jar of the glibc-floor set (package-smoke-sets.sh, check-natives.py
# floor-set) on the oldest distribution its floor promises: RHEL 8.10 -- glibc 2.28 and the system
# libstdc++.so.6 of GCC 8 -- in Red Hat's UBI 8 OpenJDK 21 runtime image, on the jar's own
# architecture. A target that is not the runner's (s390x on an x86-64 runner) runs under qemu-user
# through binfmt_misc, which the job installs (qemu-user-static); the image is multi-arch, so the same
# digest serves every row.
#
# What it proves and nothing else checks: that the dynamic loader of the old runtime accepts the
# library -- every GLIBC_x.y, GLIBCXX_x.y.z and CXXABI_x.y.z it references exists there, and so does
# every DT_NEEDED file. verify-native-deps.py compares the version numbers with the ceiling; this is
# the load itself, from the jar, as a consumer's JVM does it (LlamaLoader extracts the directory and
# System.load()s it), followed by a JNI call (NativeLoadSmoke.java, the model-free load smoke the
# macOS row runs too). The s390x jar of 6.0.0 needed GLIBC_2.38 and failed exactly here
# ("UnsatisfiedLinkError ... version `GLIBC_2.38' not found"), with every job green.
#
# The set holds every floor jar; the loader picks the one of the JVM's os.arch, and the build info it
# prints must name cpu-<target> -- a load of another jar, or none, fails. The image must report glibc
# 2.28, so a drifted image cannot turn the check into a load on a newer runtime.
#
# The smoke class is compiled on the runner (javac --release 8 against the set), so the image needs
# only a JRE.
#
# Pinned by digest: the multi-arch index (amd64, arm64, ppc64le, s390x) of ubi8/openjdk-21-runtime
# 1.23, checked 2026-10-26 -- RHEL 8.10, glibc 2.28, OpenJDK 21.0.12, uid 185. Bump it deliberately;
# it must stay an RHEL 8 image.
#
# Usage: smoke-glibc-floor.sh <set-dir> <target>
#   <set-dir>  the glibc-floor set: the classes jar, its dependencies, the floor jars
#   <target>   linux-x86-64 | linux-aarch64 | linux-s390x
set -euo pipefail

SET_DIR="${1:?usage: smoke-glibc-floor.sh <set-dir> <target>}"
TARGET="${2:?usage: smoke-glibc-floor.sh <set-dir> <target>}"
IMAGE="registry.access.redhat.com/ubi8/openjdk-21-runtime@sha256:4ce885c8f0c78dfaa11d53b12e99b4e61c02c30b65943ac3ac2e77923ce36906"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

fail() {
    echo "::error::$*" >&2
    exit 1
}

case "$TARGET" in
    linux-x86-64)  platform="linux/amd64"; machine="x86_64" ;;
    linux-aarch64) platform="linux/arm64"; machine="aarch64" ;;
    linux-s390x)   platform="linux/s390x"; machine="s390x" ;;
    *) fail "no glibc-floor platform for target '$TARGET'" ;;
esac
classifier="cpu-$TARGET"
SET_DIR="$(cd "$SET_DIR" && pwd)"
compgen -G "$SET_DIR/llama-*-$classifier.jar" >/dev/null || fail "the set in $SET_DIR has no $classifier jar"
# A foreign target needs a binfmt_misc entry for it: Ubuntu's qemu-user-static registers qemu-s390x,
# Docker Desktop plain s390x. Without one, docker run fails with "exec format error", which the image
# checks below would misreport.
if [ "$(uname -m)" != "$machine" ] && [ -e /proc/sys/fs/binfmt_misc/status ] \
   && ! compgen -G "/proc/sys/fs/binfmt_misc/*$machine*" >/dev/null; then
    fail "the runner is $(uname -m) and binfmt_misc has no entry for $machine: install qemu-user-static first"
fi

classes="$(mktemp -d)"
trap 'rm -rf "$classes"' EXIT
javac -nowarn -Xlint:-options --release 8 -cp "$SET_DIR/*" -d "$classes" "$SCRIPT_DIR/smoke/NativeLoadSmoke.java"
chmod -R a+rX "$classes"

docker pull --quiet --platform "$platform" "$IMAGE" >/dev/null
log="$(mktemp)"
# The image runs as uid 185; the set and the class are mounted read-only, the extraction goes to /tmp.
set +e
docker run --rm --platform "$platform" -v "$SET_DIR":/set:ro -v "$classes":/cls:ro "$IMAGE" \
    sh -c 'cat /etc/redhat-release; uname -m; ldd --version | head -n 1; java -version 2>&1 | head -n 1;
           exec java -cp "/cls:/set/*" NativeLoadSmoke' 2>&1 | tee "$log"
rc=${PIPESTATUS[0]}
set -e

grep -q "^Red Hat Enterprise Linux release 8\." "$log" || fail "the image is no RHEL 8 (see the first line above)"
grep -q "^$machine\$" "$log" || fail "the container does not run on $machine"
grep -qE "^ldd .* 2\.28\$" "$log" || fail "the image's glibc is not 2.28 -- the load proves nothing about the floor"
[ "$rc" -eq 0 ] || fail "$classifier does not load on RHEL 8 ($machine), exit $rc -- see the output above"
grep -q "native load smoke OK: .*classifier=$classifier[,}]" "$log" \
    || fail "the JVM did not load $classifier (no build info naming it above)"
echo "$classifier loads on RHEL 8 ($machine, glibc 2.28)"
