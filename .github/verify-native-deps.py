#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Fail when a shipped native library needs a runtime library it did not need before.

Every jllama library is ONE file with llama.cpp and ggml linked in statically, so its dynamic
dependencies are exactly what a consumer's machine must provide. A new one is a silent break on
every machine that lacks it -- the case this guards against is ggml-rpc's RDMA transport, which
upstream switches on whenever the build host has libibverbs/librdma and which would make the
library unloadable without rdma-core. It reads the dependency list straight from the file (ELF
DT_NEEDED, PE import table, Mach-O LC_LOAD_DYLIB) with the standard library only, so it runs on
any runner and checks every architecture, including the ones binutils cannot read (Windows arm64,
Mach-O).

Usage:
  verify-native-deps.py --default <resources-root>   # exact allowlist per <OS>/<ARCH>
  verify-native-deps.py --deny <dir>...               # classifier trees: only the denylist

--default checks net/ladenthin/llama/<OS>/<ARCH>/ under the root against ALLOWED below: a
dependency outside the list fails, and so does an <OS>/<ARCH> without a list (a new platform
must be listed consciously). --deny scans every native library under the directories for DENIED
names only, because GPU classifiers legitimately need their vendor runtime.

Exit codes: 0 clean, 1 violation, 2 nothing found to check.
"""

import os
import struct
import sys

# What each default-JAR library needed when this check was introduced (5.1.0 plus the RPC backend,
# which adds nothing: its sockets are libc/libSystem/WS2_32, all already present).
ALLOWED = {
    "Linux/x86_64": {"libdl.so.2", "libgomp.so.1", "libpthread.so.0", "librt.so.1", "libstdc++.so.6",
                     "libm.so.6", "libgcc_s.so.1", "libc.so.6", "ld-linux-x86-64.so.2"},
    "Linux/aarch64": {"libgomp.so.1", "libstdc++.so.6", "libm.so.6", "libgcc_s.so.1", "libc.so.6",
                      "ld-linux-aarch64.so.1"},
    "Linux/s390x": {"libstdc++.so.6", "libm.so.6", "libgcc_s.so.1", "libc.so.6", "ld64.so.1"},
    "Linux-Android/aarch64": {"liblog.so", "libm.so", "libdl.so", "libc.so", "libandroid.so"},
    "Linux-Android/x86_64": {"liblog.so", "libm.so", "libdl.so", "libc.so", "libandroid.so"},
    "Windows/x86_64": {"ws2_32.dll", "kernel32.dll", "shell32.dll", "advapi32.dll", "vcomp140.dll"},
    "Windows/x86": {"ws2_32.dll", "kernel32.dll", "shell32.dll", "advapi32.dll", "vcomp140.dll"},
    "Windows/aarch64": {"ws2_32.dll", "kernel32.dll", "shell32.dll", "advapi32.dll"},
    "Mac/aarch64": {"/usr/lib/libc++.1.dylib", "/usr/lib/libSystem.B.dylib",
                    "/System/Library/Frameworks/Foundation.framework/Versions/C/Foundation",
                    "/System/Library/Frameworks/Metal.framework/Versions/A/Metal",
                    "/System/Library/Frameworks/MetalKit.framework/Versions/A/MetalKit",
                    "/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate",
                    "/usr/lib/libobjc.A.dylib",
                    "/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation",
                    "/System/Library/Frameworks/Security.framework/Versions/A/Security",
                    # KNOWN DEFECT, allowed only so this check reports NEW dependencies: the macOS
                    # build picks up the runner's Homebrew OpenSSL, so the shipped dylib does not load
                    # on a Mac without `brew install openssl@3`. See TODO.md ("macOS dylib links
                    # Homebrew OpenSSL"); remove these two lines with the fix.
                    "/opt/homebrew/opt/openssl@3/lib/libssl.3.dylib",
                    "/opt/homebrew/opt/openssl@3/lib/libcrypto.3.dylib"},
}

# Never acceptable in any artifact: libraries a consumer cannot be expected to have.
DENIED = ("libibverbs", "librdma", "rdma.dylib", "libmlx")

LIB_NAMES = ("libjllama.so", "jllama.dll", "libjllama.dylib")


def elf_needed(data):
    if data[:4] != b"\x7fELF":
        raise ValueError("not an ELF file")
    is64 = data[4] == 2
    end = "<" if data[5] == 1 else ">"
    if is64:
        shoff = struct.unpack_from(end + "Q", data, 0x28)[0]
        shentsize, shnum = struct.unpack_from(end + "HH", data, 0x3A)
    else:
        shoff = struct.unpack_from(end + "I", data, 0x20)[0]
        shentsize, shnum = struct.unpack_from(end + "HH", data, 0x2E)
    sections = []
    for i in range(shnum):
        off = shoff + i * shentsize
        if is64:
            _, sh_type, _, _, sh_offset, sh_size, sh_link = struct.unpack_from(end + "IIQQQQI", data, off)
        else:
            _, sh_type, _, _, sh_offset, sh_size, sh_link = struct.unpack_from(end + "IIIIIII", data, off)
        sections.append((sh_type, sh_offset, sh_size, sh_link))
    out = []
    for sh_type, sh_offset, sh_size, sh_link in sections:
        if sh_type != 6:  # SHT_DYNAMIC
            continue
        strtab = sections[sh_link]
        entry = 16 if is64 else 8
        for off in range(sh_offset, sh_offset + sh_size, entry):
            tag, val = struct.unpack_from(end + ("qQ" if is64 else "iI"), data, off)
            if tag == 0:
                break
            if tag == 1:  # DT_NEEDED
                start = strtab[1] + val
                out.append(data[start:data.index(b"\0", start)].decode())
    return out


def pe_imports(data):
    if data[:2] != b"MZ":
        raise ValueError("not a PE file")
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    nsections = struct.unpack_from("<H", data, pe + 6)[0]
    opt_size = struct.unpack_from("<H", data, pe + 20)[0]
    opt = pe + 24
    magic = struct.unpack_from("<H", data, opt)[0]
    dd = opt + (112 if magic == 0x20B else 96)
    import_rva = struct.unpack_from("<I", data, dd + 8)[0]
    sec = opt + opt_size
    table = []
    for i in range(nsections):
        vsize, vaddr, rsize, raddr = struct.unpack_from("<IIII", data, sec + i * 40 + 8)
        table.append((vaddr, max(vsize, rsize), raddr))

    def to_offset(rva):
        for vaddr, size, raddr in table:
            if vaddr <= rva < vaddr + size:
                return rva - vaddr + raddr
        raise ValueError("RVA outside every section")

    out = []
    if import_rva == 0:
        return out
    off = to_offset(import_rva)
    while True:
        name_rva = struct.unpack_from("<I", data, off + 12)[0]
        if name_rva == 0:
            break
        start = to_offset(name_rva)
        out.append(data[start:data.index(b"\0", start)].decode())
        off += 20
    return out


def macho_dylibs(data):
    magic = struct.unpack_from("<I", data, 0)[0]
    if magic != 0xFEEDFACF:
        raise ValueError("not a 64-bit Mach-O file")
    ncmds = struct.unpack_from("<I", data, 16)[0]
    off = 32
    out = []
    for _ in range(ncmds):
        cmd, size = struct.unpack_from("<II", data, off)
        if cmd in (0xC, 0x80000018, 0x8000001F, 0x80000023):  # LOAD_DYLIB, WEAK, REEXPORT, UPWARD
            name_off = struct.unpack_from("<I", data, off + 8)[0]
            start = off + name_off
            out.append(data[start:data.index(b"\0", start)].decode())
        off += size
    return out


def dependencies(path):
    with open(path, "rb") as f:
        data = f.read()
    if path.endswith(".so"):
        return elf_needed(data)
    if path.endswith(".dll"):
        return pe_imports(data)
    return macho_dylibs(data)


def find_libraries(root):
    for dirpath, _, files in os.walk(root):
        for name in files:
            if name in LIB_NAMES:
                yield os.path.join(dirpath, name)


def denied(deps):
    return [d for d in deps if any(bad in d.lower() for bad in DENIED)]


def main(argv):
    if len(argv) < 3 or argv[1] not in ("--default", "--deny"):
        print(__doc__, file=sys.stderr)
        return 2
    mode, roots = argv[1], argv[2:]
    checked = 0
    failures = []
    for root in roots:
        for path in sorted(find_libraries(root)):
            deps = dependencies(path)
            checked += 1
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            print(f"{rel}: {' '.join(deps)}")
            for d in denied(deps):
                failures.append(f"{rel} needs {d}, which no consumer can be expected to have")
            if mode == "--default":
                parts = rel.split("/")
                key = "/".join(parts[-3:-1]) if len(parts) >= 3 else ""
                allowed = ALLOWED.get(key)
                if allowed is None:
                    failures.append(f"{rel}: no dependency allowlist for '{key}' -- add one to ALLOWED")
                    continue
                for d in deps:
                    if d.lower() not in {a.lower() for a in allowed}:
                        failures.append(f"{rel} needs {d}, which it did not need before (allowed: {sorted(allowed)})")
    if checked == 0:
        print(f"no native library found under {roots}", file=sys.stderr)
        return 2
    for f in failures:
        print(f"::error::{f}", file=sys.stderr)
    print(f"{checked} native libraries checked, {len(failures)} violations")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
