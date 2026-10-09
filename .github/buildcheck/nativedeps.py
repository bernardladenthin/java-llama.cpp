# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Fail when a shipped native library needs a runtime library it did not need before.

A natives jar's dynamic dependencies are exactly what a consumer's machine must provide. A new one
is a silent break on every machine that lacks it -- the case this guards against is ggml-rpc's RDMA
transport, which upstream switches on whenever the build host has libibverbs/librdma and which
would make the library unloadable without rdma-core. It reads the dependency list straight from
the file (ELF DT_NEEDED, PE import table, Mach-O LC_LOAD_DYLIB) with the standard library only, so
it runs on any runner and checks every architecture, including the ones binutils cannot read
(Windows arm64, Mach-O).

Usage (the CLI is .github/verify-native-deps.py):
  verify-native-deps.py <natives-root>

Checks every native library (.so, .dll, .dylib) under <natives-root>/.../<OS>/<ARCH>/<backend>/.
Most directories hold one file, libjllama with llama.cpp and ggml linked in statically; a
JLLAMA_CPU_VARIANTS build (CLAUDE.md "CPU variants") holds libjllama next to ggml's shared
libraries and one CPU backend module per instruction-set level, and every one of them is checked.
The CPU builds (backend cpu, metal, msvc) and the Android OpenCL build are held to the exact
allowlist in ALLOWED plus the files next to them in the same directory: a dependency outside it
fails, and so does a CPU build without a list (a new platform must be listed consciously). The other
GPU backends are checked against DENIED only, because they legitimately need their vendor runtime.
Two more checks for ELF libraries: a library that needs a sibling must find it through the run path
`$ORIGIN` and nothing else (a build-tree path would point at the CI runner), and the directories in
GLIBC_CEILING, built in a manylinux image, may not reference a glibc symbol version above the floor
they promise. Android libraries must also have every LOAD segment 16 KB aligned.
That every listed build arrived is merge-native-artifacts.sh's check.

Exit codes: 0 clean, 1 violation, 2 nothing found to check.
"""

import os
import struct
import sys

# The four OS DLLs every Windows build imports, and the dynamic UCRT.
# crypt32.dll: cpp-httplib loads the Windows certificate store for HTTPS clients
# (CertOpenSystemStoreW, `#pragma comment(lib, "crypt32.lib")` under CPPHTTPLIB_OPENSSL_SUPPORT); an
# OS component since Windows 2000. BoringSSL itself adds no import: its RNG loads bcryptprimitives
# at run time (LoadLibraryW), and everything else is in the static libraries.
WINDOWS_OS = {"ws2_32.dll", "kernel32.dll", "shell32.dll", "advapi32.dll", "crypt32.dll"}
# The Universal CRT forwarders. Present because the Windows builds use the HYBRID CRT (static STL +
# vcruntime, dynamic UCRT -- llama/CMakeLists.txt explains why), which is what keeps Microsoft's
# UCRT security updates reaching a shipped artifact instead of freezing a copy inside jllama.dll.
# They are an OS component from Windows 10 on, which README.md states as the requirement.
# Note what is NOT here and must not be added: msvcp140.dll / vcruntime140.dll (the static half of
# the hybrid -- their appearance would mean the CRT went dynamic) and vcomp140.dll (MSVC's OpenMP:
# every Windows CPU job passes -DGGML_OPENMP=OFF, a measured ~2x on token generation, so this list
# is the guard that fails the build if that flag is dropped).
WINDOWS_UCRT_PREFIX = "api-ms-win-crt-"


def ucrt_forwarder(name):
    """Whether `name` is a Universal CRT forwarder, i.e. evidence of a DYNAMIC UCRT.

    Matched by PREFIX, not by an enumerated list, and that is the point: the invariant worth
    guarding is "the UCRT is the operating system's", and any api-ms-win-crt-* import proves it.
    Which of the ~11 forwarders a given library ends up importing depends on the CRT functions it
    happens to use, so it legitimately differs between x86-64 and x86, between cl.exe, clang-cl and
    plain clang, and after any upstream change that calls one more CRT function. Enumerating them
    would add no safety whatsoever -- a static UCRT shows up as *none* of them, which the
    WINDOWS_UCRT_REQUIRED check below catches either way -- while turning every such difference
    into a red `package` job. (An earlier version did enumerate eleven names, measured on one
    plain-clang build.)"""
    return name.lower().startswith(WINDOWS_UCRT_PREFIX)
# Where at least one of those must be PRESENT, which is the other direction of the same guard --
# see the check in violations(). The GPU directories are left out: they are held to a denylist, and
# a vendor toolchain may link its own runtime.
WINDOWS_UCRT_REQUIRED = {"Windows/x86_64/cpu", "Windows/x86/cpu", "Windows/aarch64/cpu",
                         "Windows/x86_64/msvc", "Windows/x86/msvc"}

# What each CPU library needed when this check was introduced (5.1.0 plus the RPC backend,
# which adds nothing: its sockets are libc/libSystem/WS2_32, all already present). The manylinux_2_28
# builds (glibc 2.28, before the libpthread/libdl/librt merge of 2.34) name those three separately.
ALLOWED = {
    "Linux/x86_64/cpu": {"libdl.so.2", "libgomp.so.1", "libpthread.so.0", "librt.so.1", "libstdc++.so.6",
                     "libm.so.6", "libgcc_s.so.1", "libc.so.6", "ld-linux-x86-64.so.2"},
    "Linux/aarch64/cpu": {"libdl.so.2", "libgomp.so.1", "libpthread.so.0", "librt.so.1", "libstdc++.so.6",
                      "libm.so.6", "libgcc_s.so.1", "libc.so.6", "ld-linux-aarch64.so.1"},
    "Linux/s390x/cpu": {"libstdc++.so.6", "libm.so.6", "libgcc_s.so.1", "libc.so.6", "ld64.so.1"},
    "Linux-Android/aarch64/cpu": {"liblog.so", "libm.so", "libdl.so", "libc.so", "libandroid.so"},
    "Linux-Android/x86_64/cpu": {"liblog.so", "libm.so", "libdl.so", "libc.so", "libandroid.so"},
    # Exactly WINDOWS_OS, plus any api-ms-win-crt-* forwarder through ucrt_forwarder() -- see
    # WINDOWS_OS above for what is deliberately absent (msvcp140/vcruntime140 and vcomp140) and why
    # that makes this list a guard, and WINDOWS_UCRT_REQUIRED below for the other direction.
    "Windows/x86_64/cpu": WINDOWS_OS,
    "Windows/x86/cpu": WINDOWS_OS,
    "Windows/aarch64/cpu": WINDOWS_OS,
    "Mac/aarch64/metal": {"/usr/lib/libc++.1.dylib", "/usr/lib/libSystem.B.dylib",
                    "/System/Library/Frameworks/Foundation.framework/Versions/C/Foundation",
                    "/System/Library/Frameworks/Metal.framework/Versions/A/Metal",
                    "/System/Library/Frameworks/MetalKit.framework/Versions/A/MetalKit",
                    "/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate",
                    "/usr/lib/libobjc.A.dylib",
                    "/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation",
                    # Security + CoreFoundation: cpp-httplib's system-certificate lookup under
                    # CPPHTTPLIB_OPENSSL_SUPPORT. The SSL library itself is BoringSSL, linked
                    # statically (llama/CMakeLists.txt, "HTTPS"); a /opt/homebrew/.../libssl line
                    # here would mean the build picked up the runner's OpenSSL again.
                    "/System/Library/Frameworks/Security.framework/Versions/A/Security"},
}
# The Visual Studio generator build of the same compiler and runtime.
ALLOWED["Windows/x86_64/msvc"] = ALLOWED["Windows/x86_64/cpu"]
ALLOWED["Windows/x86/msvc"] = ALLOWED["Windows/x86/cpu"]
# The OpenCL AAR flavour: an app bundles no other native library, so the Android GPU build is held
# to an exact list as well -- the bionic system libraries plus the vendor ICD. (libomp.so and
# libc++_shared.so once shipped exactly this way and failed System.loadLibrary on every device.)
ALLOWED["Linux-Android/aarch64/opencl"] = ALLOWED["Linux-Android/aarch64/cpu"] | {"libOpenCL.so"}

# The glibc floor a directory promises (README, "Runtime requirement"): the highest GLIBC_x.y symbol
# version any of its libraries references may not exceed it. These are the builds made in a
# manylinux_2_28 image (publish.yml: crosscompile-linux-x86_64, crosscompile-linux-aarch64,
# crosscompile-linux-x86_64-cuda); the other Linux builds inherit the floor of the ubuntu runner.
GLIBC_CEILING = {
    "Linux/x86_64/cpu": (2, 28),
    "Linux/aarch64/cpu": (2, 28),
    "Linux/x86_64/cuda13": (2, 28),
}

# Google Play's 16 KB page-size requirement (Android 15+ targets): every LOAD segment of an
# Android library must be aligned to a multiple of it. CMake pins -Wl,-z,max-page-size=16384.
ANDROID_PAGE_ALIGNMENT = 16384
CPU_BACKENDS = ("cpu", "metal", "msvc")
LIBRARY_SUFFIXES = (".so", ".dll", ".dylib")
ORIGIN = "$ORIGIN"

# Never acceptable in any artifact: libraries a consumer cannot be expected to have.
DENIED = ("libibverbs", "librdma", "rdma.dylib", "libmlx")

DT_NEEDED, DT_RPATH, DT_RUNPATH = 1, 15, 29
SHT_DYNAMIC, SHT_GNU_VERNEED = 6, 0x6FFFFFFE


def _elf_sections(data):
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
            _, sh_type, _, _, sh_offset, sh_size, sh_link, sh_info = struct.unpack_from(end + "IIQQQQII", data, off)
        else:
            _, sh_type, _, _, sh_offset, sh_size, sh_link, sh_info = struct.unpack_from(end + "IIIIIIII", data, off)
        sections.append((sh_type, sh_offset, sh_size, sh_link, sh_info))
    return is64, end, sections


def _cstring(data, start):
    return data[start:data.index(b"\0", start)].decode()


def elf_info(data):
    """DT_NEEDED, the run path (DT_RUNPATH, else DT_RPATH, else None) and the highest GLIBC_x.y
    symbol version the library references (a tuple, or None when it references none)."""
    is64, end, sections = _elf_sections(data)
    needed, runpath, rpath, glibc = [], None, None, []
    for sh_type, sh_offset, sh_size, sh_link, sh_info in sections:
        strtab = sections[sh_link][1] if sh_link < len(sections) else 0
        if sh_type == SHT_DYNAMIC:
            entry = 16 if is64 else 8
            for off in range(sh_offset, sh_offset + sh_size, entry):
                tag, val = struct.unpack_from(end + ("qQ" if is64 else "iI"), data, off)
                if tag == 0:
                    break
                if tag == DT_NEEDED:
                    needed.append(_cstring(data, strtab + val))
                elif tag == DT_RUNPATH:
                    runpath = _cstring(data, strtab + val)
                elif tag == DT_RPATH:
                    rpath = _cstring(data, strtab + val)
        elif sh_type == SHT_GNU_VERNEED:
            off = sh_offset
            for _ in range(sh_info):
                _, vn_cnt, _, vn_aux, vn_next = struct.unpack_from(end + "HHIII", data, off)
                aux = off + vn_aux
                for _ in range(vn_cnt):
                    _, _, _, vna_name, vna_next = struct.unpack_from(end + "IHHII", data, aux)
                    name = _cstring(data, strtab + vna_name)
                    if name.startswith("GLIBC_"):
                        glibc.append(tuple(int(p) for p in name[len("GLIBC_"):].split(".")))
                    aux += vna_next
                if vn_next == 0:
                    break
                off += vn_next
    return {"needed": needed, "runpath": runpath if runpath is not None else rpath,
            "glibc": max(glibc) if glibc else None}


def elf_needed(data):
    return elf_info(data)["needed"]


def elf_load_alignments(data):
    """p_align of every PT_LOAD program header."""
    if data[:4] != b"\x7fELF":
        raise ValueError("not an ELF file")
    is64 = data[4] == 2
    end = "<" if data[5] == 1 else ">"
    if is64:
        phoff = struct.unpack_from(end + "Q", data, 0x20)[0]
        phentsize, phnum = struct.unpack_from(end + "HH", data, 0x36)
    else:
        phoff = struct.unpack_from(end + "I", data, 0x1C)[0]
        phentsize, phnum = struct.unpack_from(end + "HH", data, 0x2A)
    out = []
    for i in range(phnum):
        off = phoff + i * phentsize
        if struct.unpack_from(end + "I", data, off)[0] == 1:  # PT_LOAD
            out.append(struct.unpack_from(end + ("Q" if is64 else "I"), data, off + (0x30 if is64 else 0x1C))[0])
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
        out.append(_cstring(data, start))
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
            out.append(_cstring(data, off + name_off))
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
            if name.endswith(LIBRARY_SUFFIXES):
                yield os.path.join(dirpath, name)


def denied(deps):
    return [d for d in deps if any(bad in d.lower() for bad in DENIED)]


def glibc_name(version):
    return "GLIBC_" + ".".join(str(p) for p in version)


def violations(rel, deps, alignments=(), siblings=(), runpath=None, glibc=None):
    """The violations of one library, `rel` being its path below the natives root
    (.../<OS>/<ARCH>/<backend>/<library>); `alignments` are its LOAD segment alignments (checked
    for Android libraries), `siblings` the names of the other files in its directory, `runpath`
    its ELF run path and `glibc` the highest glibc symbol version it references."""
    failures = [f"{rel} needs {d}, which no consumer can be expected to have" for d in denied(deps)]
    parts = rel.split("/")
    key = "/".join(parts[-4:-1]) if len(parts) >= 4 else ""
    if key.startswith("Linux-Android/"):
        failures += [f"{rel}: LOAD alignment {a} is not a multiple of {ANDROID_PAGE_ALIGNMENT} "
                     f"(Google Play 16 KB page-size requirement)" for a in alignments if a % ANDROID_PAGE_ALIGNMENT]
    siblings = set(siblings)
    if rel.endswith(".so") and any(d in siblings for d in deps) and runpath != ORIGIN:
        failures.append(f"{rel} needs the sibling {sorted(d for d in deps if d in siblings)} but its run path is "
                        f"{runpath!r}, not {ORIGIN!r} -- it would be looked up on the system instead")
    # The allowlist below only reports dependencies that should NOT be there. For the Windows CPU
    # directories one dependency must be there, and its disappearance is just as much a regression:
    # an api-ms-win-crt-* forwarder means the UCRT is the OS one (hybrid CRT). Without this, a build
    # that silently went back to a fully static CRT -- a plain /MT, which freezes a copy of the UCRT
    # inside the library so Microsoft's security updates never reach it -- would pass unnoticed,
    # because a static CRT only ever REMOVES imports. Measured on a real variants build: all 18
    # libraries import between 5 and 11 of these, so "at least one" holds for modules too.
    if key in WINDOWS_UCRT_REQUIRED and not any(ucrt_forwarder(d) for d in deps):
        failures.append(f"{rel} imports no api-ms-win-crt-* forwarder, so its UCRT is statically linked. "
                        f"The Windows builds use the hybrid CRT on purpose (llama/CMakeLists.txt): a static "
                        f"UCRT cannot receive Microsoft's security updates. Was a linker flag lost?")
    ceiling = GLIBC_CEILING.get(key)
    if ceiling and glibc and glibc > ceiling:
        failures.append(f"{rel} references {glibc_name(glibc)}, above the {glibc_name(ceiling)} floor its "
                        f"directory promises -- was it built in the manylinux image?")
    allowed = ALLOWED.get(key)
    if allowed is None:
        if key.rsplit("/", 1)[-1] not in CPU_BACKENDS:
            return failures
        return failures + [f"{rel}: no dependency allowlist for '{key}' -- add one to ALLOWED"]
    lowered = {a.lower() for a in allowed} | {s.lower() for s in siblings}
    return failures + [f"{rel} needs {d}, which it did not need before (allowed: {sorted(allowed)}"
                       f"{' + the files next to it' if siblings else ''})"
                       for d in deps if d.lower() not in lowered and not ucrt_forwarder(d)]


def main(argv):
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    root = argv[1]
    checked = 0
    failures = []
    for path in sorted(find_libraries(root)):
        checked += 1
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        siblings = {n for n in os.listdir(os.path.dirname(path)) if n != os.path.basename(path)}
        alignments, runpath, glibc, note = (), None, None, ""
        try:
            if path.endswith(".so"):
                with open(path, "rb") as f:
                    data = f.read()
                info = elf_info(data)
                deps, runpath, glibc = info["needed"], info["runpath"], info["glibc"]
                if "/Linux-Android/" in "/" + rel:
                    alignments = elf_load_alignments(data)
                note = (f" [runpath={runpath}]" if runpath else "") + (f" [{glibc_name(glibc)}]" if glibc else "")
            else:
                deps = dependencies(path)
        except (ValueError, struct.error) as e:
            failures.append(f"{rel}: not a readable native library ({e})")
            continue
        print(f"{rel}: {' '.join(deps)}{note}")
        failures += violations(rel, deps, alignments, siblings, runpath, glibc)
    if checked == 0:
        print(f"no native library found under {root}", file=sys.stderr)
        return 2
    for f in failures:
        print(f"::error::{f}", file=sys.stderr)
    print(f"{checked} native libraries checked, {len(failures)} violations")
    return 1 if failures else 0
