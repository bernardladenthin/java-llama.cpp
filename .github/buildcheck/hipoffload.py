# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Fail when a ROCm/HIP build ships its GPU code uncompressed.

llama/CMakeLists.txt compiles ggml-hip with clang's --offload-compress, so every embedded
device-code bundle is a compressed "CCOB" bundle. An uncompressed bundle starts with the magic
"__CLANG_OFFLOAD_BUNDLE__"; one of those in the shipped library means the flag was lost and the
library is back to carrying every GPU target's code uncompressed (~1 GB on Windows).

Usage (the CLI is .github/verify-hip-offload-compressed.py): verify-hip-offload-compressed.py <dir-or-file>...

Scans every ggml-hip.dll / libggml-hip.so found (the ROCm module of a JLLAMA_MODULE_ONLY build), prints its size and the bundle counts, and exits
non-zero if a library has an uncompressed bundle, has no compressed bundle at all, or if no
library was found. Only the standard library, so it runs on any runner.
"""

import os
import sys

UNCOMPRESSED = b"__CLANG_OFFLOAD_BUNDLE__"
COMPRESSED = b"CCOB"
NAMES = ("ggml-hip.dll", "libggml-hip.so")
CHUNK = 64 * 1024 * 1024


def count(path, chunk=CHUNK):
    """Counts both magics in one streaming pass (the file can be ~1 GB)."""
    overlap = len(UNCOMPRESSED) - 1
    uncompressed = compressed = 0
    tail = b""
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            data = tail + block
            # a match lying entirely inside `tail` was counted in the previous round
            uncompressed += data.count(UNCOMPRESSED) - tail.count(UNCOMPRESSED)
            compressed += data.count(COMPRESSED) - tail.count(COMPRESSED)
            tail = data[-overlap:]
    return uncompressed, compressed


def libraries(paths):
    for p in paths:
        if os.path.isfile(p):
            yield p
        for root, _, files in os.walk(p):
            for name in files:
                if name in NAMES:
                    yield os.path.join(root, name)


def main(argv):
    if len(argv) < 2:
        print("usage: verify-hip-offload-compressed.py <dir-or-file>...", file=sys.stderr)
        return 2
    found = failed = 0
    summary = []
    for lib in libraries(argv[1:]):
        found += 1
        size = os.path.getsize(lib)
        uncompressed, compressed = count(lib)
        line = f"{lib}: {size / 1024 / 1024:.0f} MiB, {compressed} compressed / {uncompressed} uncompressed offload bundle(s)"
        print(line)
        summary.append(line)
        if uncompressed:
            print(f"::error::{lib} embeds {uncompressed} uncompressed GPU code bundle(s); is --offload-compress still set on ggml-hip?")
            failed += 1
        elif not compressed:
            print(f"::error::{lib} embeds no compressed GPU code bundle; was it built with GGML_HIP=ON?")
            failed += 1
    if not found:
        print(f"::error::no {' / '.join(NAMES)} found under {argv[1:]}")
        return 2
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as f:
            f.write("### ROCm/HIP device code\n\n" + "\n".join(f"- `{s}`" for s in summary) + "\n")
    return 1 if failed else 0
