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
  verify-native-deps.py [--require-signed] <natives-root>

--require-signed: every Windows DLL must be Authenticode-signed (jllama-build.txt says signed=yes
and the file carries a signature); without it only a directory saying signed=yes is checked.

Checks every native library (.so, .dll, .dylib) under <natives-root>/.../<OS>/<ARCH>/<backend>/, and
the export set of every jllama library (JNI entry points and the jllama_* C API only).
A library directory (backend cpu or metal) holds libjllama -- alone where llama.cpp and ggml are
linked in statically (macOS, s390x), next to ggml's shared libraries and one CPU backend module per
instruction-set level in a JLLAMA_BACKEND_DL build (CLAUDE.md "Modular natives") -- and every one of
them is checked. The library directories and the Android OpenCL module are held to the exact
allowlist in ALLOWED plus the files next to them in the same directory: a dependency outside it
fails, and so does a library directory without a list (a new platform must be listed consciously).
A module directory (every other backend) holds one GPU backend module that LlamaLoader puts next to
the library jar's files: it is checked against DENIED only, because it legitimately needs its vendor
runtime, and it must import ggml-base -- the proof that it is a module built against the shared ggml
and not a second monolithic library. Two more checks for ELF libraries: one that needs a sibling,
or is a module (whose libggml-base lies next to it only after extraction), must carry the run path
`$ORIGIN` and nothing else (a build-tree path would point at the CI runner) -- except on Android,
where a library carries no run path at all and only a build-tree path fails (see violations()) --
and the directories in GLIBC_CEILING, built in a manylinux image or against its sysroot, may not
reference a glibc symbol version above the floor they promise, nor a libstdc++ one (GLIBCXX_*,
CXXABI_*) above LIBSTDCXX_CEILING. Android libraries must also have every LOAD segment 16 KB
aligned, and on s390x every typeinfo name (_ZTS*) of the static symbol table must sit at the even
address the ABI promises (TYPEINFO_NAME_ALIGNMENT; a jllama library without any is not examined,
which fails too).
That every listed build arrived is merge-native-artifacts.sh's check.

Exit codes: 0 clean, 1 violation, 2 nothing found to check.
"""

import os
import re
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
WINDOWS_UCRT_REQUIRED = {"Windows/x86_64/cpu", "Windows/aarch64/cpu"}

# What each CPU library needed when this check was introduced (5.1.0 plus the RPC backend,
# which adds nothing: its sockets are libc/libSystem/WS2_32, all already present). The manylinux_2_28
# builds (glibc 2.28, before the libpthread/libdl/librt merge of 2.34) name those three separately;
# s390x is one of them since its build moved to the manylinux_2_28 sysroot (2026-10-25); its modular
# directory (6.1.0) needs libpthread (libjllama, libmtmd, libggml-base) and libdl (libggml), not
# librt (measured), as did the single static library before it.
# libgomp.so.1 is deliberately absent from the three Linux cpu entries: llama/CMakeLists.txt forces
# GGML_OPENMP=OFF there so the jar loads in slim and distroless images, which do not ship it, and
# this list is what keeps it that way -- a build that regains the dependency fails `package` instead
# of shipping a jar that cannot load. libstdc++/libgcc_s stay listed on purpose: every image that
# can run this library has them, because HotSpot's libjvm.so links them too.
ALLOWED = {
    "Linux/x86_64/cpu": {"libdl.so.2", "libpthread.so.0", "librt.so.1", "libstdc++.so.6",
                     "libm.so.6", "libgcc_s.so.1", "libc.so.6", "ld-linux-x86-64.so.2"},
    "Linux/aarch64/cpu": {"libdl.so.2", "libpthread.so.0", "librt.so.1", "libstdc++.so.6",
                      "libm.so.6", "libgcc_s.so.1", "libc.so.6", "ld-linux-aarch64.so.1"},
    "Linux/s390x/cpu": {"libdl.so.2", "libpthread.so.0", "libstdc++.so.6", "libm.so.6", "libgcc_s.so.1",
                        "libc.so.6", "ld64.so.1"},
    "Linux-Android/aarch64/cpu": {"liblog.so", "libm.so", "libdl.so", "libc.so", "libandroid.so"},
    "Linux-Android/x86_64/cpu": {"liblog.so", "libm.so", "libdl.so", "libc.so", "libandroid.so"},
    # Exactly WINDOWS_OS, plus any api-ms-win-crt-* forwarder through ucrt_forwarder() -- see
    # WINDOWS_OS above for what is deliberately absent (msvcp140/vcruntime140 and vcomp140) and why
    # that makes this list a guard, and WINDOWS_UCRT_REQUIRED below for the other direction.
    "Windows/x86_64/cpu": WINDOWS_OS,
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
# The OpenCL AAR flavour: an app bundles no other native library, so the Android GPU module is held
# to an exact list as well -- the bionic system libraries, the vendor ICD and the ggml-base it is a
# module of (the CPU AAR's, resolved by soname in the app's linker namespace). (libomp.so and
# libc++_shared.so once shipped exactly this way and failed System.loadLibrary on every device.)
ALLOWED["Linux-Android/aarch64/opencl"] = ALLOWED["Linux-Android/aarch64/cpu"] | {"libOpenCL.so", "libggml-base.so"}

# The glibc floor a directory promises (README, "Runtime requirement"): the highest GLIBC_x.y symbol
# version any of its libraries references may not exceed it. These are the builds made in a
# manylinux_2_28 image (publish.yml: crosscompile-linux-x86_64, crosscompile-linux-aarch64,
# crosscompile-linux-x86_64-cuda) or cross-compiled against one unpacked as the sysroot
# (build-linux-s390x, .github/manylinux/s390x-toolchain.cmake); the other Linux builds inherit the
# floor of the ubuntu runner. s390x joined on 2026-10-25: its cross build against the ubuntu
# runner's own sysroot had shipped GLIBC_2.38 without any check noticing (CLAUDE.md, "Linux s390x").
GLIBC_CEILING = {
    "Linux/x86_64/cpu": (2, 28),
    "Linux/aarch64/cpu": (2, 28),
    "Linux/x86_64/cuda13": (2, 28),
    "Linux/s390x/cpu": (2, 28),
}
# The libstdc++ half of the same floor, for the same directories: the system libstdc++.so.6 of
# manylinux_2_28 is GCC 8's (GLIBCXX_3.4.25, CXXABI_1.3.11), and gcc-toolset-14 links whatever its
# GCC 14 headers need beyond that from libstdc++_nonshared.a. A version above it means the library
# was linked against a newer libstdc++ and fails to load on RHEL 8 even when its glibc symbols fit
# (the s390x jar of 6.0.0 needed GLIBCXX_3.4.32 and CXXABI_1.3.13). Measured on the CI artifacts of
# 2026-10-25: x86-64 and aarch64 reference 3.4.25 / 1.3.11 exactly, the CUDA module 3.4.21 / 1.3.9.
LIBSTDCXX_CEILING = {"GLIBCXX": (3, 4, 25), "CXXABI": (1, 3, 11)}
# The symbol versions elf_info() reports, by family. libstdc++ on s390x (and ppc64) also defines
# GLIBCXX_LDBL_x / CXXABI_LDBL_x for its second long double ABI; they count as their family.
# Anything else (GLIBC_PRIVATE, CXXABI_TM_1, ...) carries no floor and is skipped.
SYMBOL_VERSION = re.compile(r"(GLIBC|GLIBCXX|CXXABI)_(?:LDBL_)?(\d+(?:\.\d+)*)")

# Google Play's 16 KB page-size requirement (Android 15+ targets): every LOAD segment of an
# Android library must be aligned to a multiple of it. CMake pins -Wl,-z,max-page-size=16384.
ANDROID_PAGE_ALIGNMENT = 16384
CPU_BACKENDS = ("cpu", "metal")
LIBRARY_SUFFIXES = (".so", ".dll", ".dylib")
ORIGIN = "$ORIGIN"
# What a GPU module must import: the shared ggml-base of the library jar it is extracted next to.
GGML_BASE = ("libggml-base.so", "ggml-base.dll")

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
    """DT_NEEDED, the run path (DT_RUNPATH, else DT_RPATH, else None) and the highest GLIBC_x.y,
    GLIBCXX_x.y.z and CXXABI_x.y.z symbol version the library references (`glibc`, `glibcxx`,
    `cxxabi`: a tuple each, or None when it references none of that family)."""
    is64, end, sections = _elf_sections(data)
    needed, runpath, rpath = [], None, None
    versions = {"GLIBC": [], "GLIBCXX": [], "CXXABI": []}
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
                    match = SYMBOL_VERSION.fullmatch(_cstring(data, strtab + vna_name))
                    if match:
                        versions[match.group(1)].append(tuple(int(p) for p in match.group(2).split(".")))
                    aux += vna_next
                if vn_next == 0:
                    break
                off += vn_next
    top = {family: max(found) if found else None for family, found in versions.items()}
    return {"needed": needed, "runpath": runpath if runpath is not None else rpath,
            "glibc": top["GLIBC"], "glibcxx": top["GLIBCXX"], "cxxabi": top["CXXABI"]}


def elf_needed(data):
    return elf_info(data)["needed"]


SHT_DYNSYM = 11
STB_GLOBAL, STB_WEAK = 1, 2


def elf_exports(data):
    """The defined global and weak symbols of the dynamic symbol table (.dynsym), i.e. what a
    consumer of the shared object can resolve."""
    is64, end, sections = _elf_sections(data)
    out = []
    for sh_type, sh_offset, sh_size, sh_link, _ in sections:
        if sh_type != SHT_DYNSYM:
            continue
        strtab = sections[sh_link][1]
        entry = 24 if is64 else 16
        for off in range(sh_offset, sh_offset + sh_size, entry):
            if is64:
                st_name, st_info, _, st_shndx = struct.unpack_from(end + "IBBH", data, off)
            else:
                st_name, _, _, st_info, _, st_shndx = struct.unpack_from(end + "IIIBBH", data, off)
            if st_shndx == 0 or (st_info >> 4) not in (STB_GLOBAL, STB_WEAK):
                continue
            name = _cstring(data, strtab + st_name)
            if name:
                out.append(name)
    return out


SHT_SYMTAB = 2
SHN_LORESERVE = 0xFF00


def elf_symbols(data):
    """(name, address) of every defined symbol of the static symbol table (.symtab), whatever its
    binding: the export list and -fvisibility=hidden make almost every symbol of a shipped library
    local there. A stripped library has no .symtab, and the list is empty."""
    is64, end, sections = _elf_sections(data)
    out = []
    for sh_type, sh_offset, sh_size, sh_link, _ in sections:
        if sh_type != SHT_SYMTAB:
            continue
        strtab = sections[sh_link][1]
        entry = 24 if is64 else 16
        for off in range(sh_offset, sh_offset + sh_size, entry):
            if is64:
                st_name, _, _, st_shndx, st_value = struct.unpack_from(end + "IBBHQ", data, off)
            else:
                st_name, st_value, _, _, _, st_shndx = struct.unpack_from(end + "IIIBBH", data, off)
            if st_name == 0 or st_shndx == 0 or st_shndx >= SHN_LORESERVE:
                continue
            out.append((_cstring(data, strtab + st_name), st_value))
    return out


# The s390x ELF ABI gives every global symbol an even address, and GCC relies on it: it loads an
# address with LARL, whose R_390_PC32DBL relocation counts in halfwords. clang emits typeinfo name
# strings (_ZTS*) with the alignment of char (ItaniumCXXABI.cpp, GetAddrOfTypeName), and the
# s390x build mixes clang's objects with GCC's -- gcc-toolset-14's libstdc++_nonshared.a, which
# loads _ZTSSt19_Sp_make_shared_tag with LARL. Whether a link then fails ("misaligned symbol")
# depends on where the copy it keeps lands: green on main, red in the next PR's jllama_test link
# (#577). .github/manylinux/s390x-typeinfo-names.ld (SUBALIGN(2) over every .rodata._ZTS*
# section) and -fdata-sections (which gives every typeinfo name such a section, the ones of
# classes with a key function included) make it hold for all of them; this check is what notices
# when they stop being in effect, in every build rather than in the one whose layout comes out odd.
TYPEINFO_NAME_ALIGNMENT = {"s390x": 2}


def typeinfo_name_alignment(rel):
    """The alignment the ABI of the directory of `rel` (.../<OS>/<ARCH>/<backend>/<library>)
    requires for typeinfo names, or None."""
    parts = rel.split("/")
    return TYPEINFO_NAME_ALIGNMENT.get(parts[-3]) if len(parts) >= 4 else None


def typeinfo_name_violations(rel, symbols, alignment, required=True):
    """Typeinfo names (_ZTS*) among `symbols` ((name, address) pairs) at an address that is not a
    multiple of `alignment`. A library that must carry them (`required`) and carries none was not
    examined -- stripped, or its table unreadable -- which is never a pass."""
    names = [(n, v) for n, v in symbols if n.startswith("_ZTS")]
    if not names:
        if not required:
            return []
        return [f"{rel}: no typeinfo name (_ZTS*) in its symbol table ({len(symbols)} symbols) -- stripped, or the "
                f"table could not be read, so their {alignment}-byte alignment cannot be checked"]
    odd = [n for n, v in names if v % alignment]
    if not odd:
        return []
    shown = ", ".join(odd[:8]) + (f", ... ({len(odd)} in all)" if len(odd) > 8 else "")
    return [f"{rel}: {len(odd)} of {len(names)} typeinfo names (_ZTS*) are not {alignment}-byte aligned: {shown} -- "
            f"GCC code (gcc-toolset's libstdc++_nonshared.a) loads them with LARL, so a link that keeps such a "
            f"copy fails with 'misaligned symbol'; .github/manylinux/s390x-typeinfo-names.ld or -fdata-sections "
            f"(s390x-toolchain.cmake) is not in effect"]


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


def _pe_layout(data):
    """(data directory offset, rva -> file offset) of a PE32/PE32+ image."""
    if data[:2] != b"MZ":
        raise ValueError("not a PE file")
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    nsections = struct.unpack_from("<H", data, pe + 6)[0]
    opt_size = struct.unpack_from("<H", data, pe + 20)[0]
    opt = pe + 24
    magic = struct.unpack_from("<H", data, opt)[0]
    dd = opt + (112 if magic == 0x20B else 96)
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

    return dd, to_offset


def pe_signed(data):
    """Whether the PE carries an Authenticode signature: IMAGE_DIRECTORY_ENTRY_SECURITY (index 4)
    names a non-empty attribute certificate table. Presence only -- the chain is verified by the
    signer's own tooling and by Windows; a stripped or never-signed file has a zero entry."""
    dd, _ = _pe_layout(data)
    address, size = struct.unpack_from("<II", data, dd + 4 * 8)
    return address != 0 and size != 0


def signed_state(directory):
    """The `signed=` value of a natives directory's jllama-build.txt, or None without the file."""
    build = os.path.join(directory, "jllama-build.txt")
    if not os.path.isfile(build):
        return None
    with open(build, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.startswith("signed="):
                return line[len("signed="):]
    return None


def signature_violations(root, require_signed=False):
    """The Windows directories whose signature state and DLLs disagree.

    A directory whose jllama-build.txt says `signed=yes` must hold only Authenticode-signed DLLs
    (the sign step flips the key after signing every DLL, so a `yes` over an unsigned file means a
    file was added or replaced afterwards). With `require_signed`, a Windows directory that says
    `signed=no` (or carries no build file) is a violation too: the release rule once the owner's
    signing account exists, so that no unsigned DLL ever reaches a release jar."""
    failures = []
    for path in sorted(find_libraries(root)):
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        if "/Windows/" not in "/" + rel or not path.lower().endswith(".dll"):
            continue
        state = signed_state(os.path.dirname(path))
        if state == "yes":
            with open(path, "rb") as f:
                data = f.read()
            try:
                if not pe_signed(data):
                    failures.append(f"{rel}: jllama-build.txt says signed=yes but the DLL carries no Authenticode signature")
            except (ValueError, struct.error) as e:
                failures.append(f"{rel}: not a readable PE ({e})")
        elif require_signed:
            failures.append(f"{rel}: unsigned (jllama-build.txt says signed={state}) while a signature is required")
    return failures


def pe_exports(data):
    """The names of the export directory (IMAGE_DIRECTORY_ENTRY_EXPORT)."""
    dd, to_offset = _pe_layout(data)
    export_rva, export_size = struct.unpack_from("<II", data, dd)
    if export_rva == 0 or export_size == 0:
        return []
    directory = to_offset(export_rva)
    number_of_names = struct.unpack_from("<I", data, directory + 24)[0]
    names_rva = struct.unpack_from("<I", data, directory + 32)[0]
    names = to_offset(names_rva)
    return [_cstring(data, to_offset(struct.unpack_from("<I", data, names + 4 * i)[0])) for i in range(number_of_names)]


N_EXT, N_PEXT, N_TYPE, N_SECT, N_STAB = 0x01, 0x10, 0x0E, 0x0E, 0xE0


def macho_exports(data):
    """The external, defined, non-private symbols of the symbol table (LC_SYMTAB) of a 64-bit
    Mach-O, with their leading underscore kept as the file carries it."""
    magic = struct.unpack_from("<I", data, 0)[0]
    if magic != 0xFEEDFACF:
        raise ValueError("not a 64-bit Mach-O file")
    ncmds = struct.unpack_from("<I", data, 16)[0]
    off = 32
    out = []
    for _ in range(ncmds):
        cmd, size = struct.unpack_from("<II", data, off)
        if cmd == 0x2:  # LC_SYMTAB
            symoff, nsyms, stroff, _ = struct.unpack_from("<IIII", data, off + 8)
            for i in range(nsyms):
                n_strx, n_type, _, _, _ = struct.unpack_from("<IBBHQ", data, symoff + 16 * i)
                if n_type & N_STAB or not n_type & N_EXT or n_type & N_PEXT or (n_type & N_TYPE) != N_SECT:
                    continue
                name = _cstring(data, stroff + n_strx)
                if name:
                    out.append(name)
        off += size
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


def exports(path):
    with open(path, "rb") as f:
        data = f.read()
    if path.endswith(".so"):
        return elf_exports(data)
    if path.endswith(".dll"):
        return pe_exports(data)
    return [n[1:] if n.startswith("_") else n for n in macho_exports(data)]


# The jllama library exports the JNI entry points the JVM resolves and the jllama_* C API, nothing
# else (llama/cmake/jllama-exports.map): everything it links statically -- llama, llama-common,
# mtmd, cpp-httplib, BoringSSL, ggml in a static build -- stays local. Before the export list the
# Linux library exported 14,220 symbols, 47 of them JNI: an accidental C++ ABI with no header
# contract, which a consumer could link against and which every llama.cpp bump would break.
JLLAMA_LIBRARIES = ("libjllama.so", "jllama.dll", "libjllama.dylib")
EXPORT_ALLOWED = ("JNI_OnLoad", "JNI_OnUnload")
EXPORT_PREFIXES = ("Java_", "jllama_")


def export_violations(rel, names):
    """Why the export set of a jllama library is wrong: a symbol outside the allowed set, or no
    JNI entry point at all (an empty list means the file is not a loadable library, or the parser
    read nothing -- never a pass)."""
    if not any(n.startswith("Java_") for n in names):
        return [f"{rel}: exports no Java_* entry point ({len(names)} exported symbols) -- not a JNI library, or the "
                f"symbol table could not be read"]
    stray = [n for n in names if n not in EXPORT_ALLOWED and not n.startswith(EXPORT_PREFIXES)]
    if not stray:
        return []
    shown = ", ".join(stray[:8]) + (f", ... ({len(stray)} in all)" if len(stray) > 8 else "")
    return [f"{rel}: exports {len(stray)} symbol(s) outside JNI_OnLoad/JNI_OnUnload/Java_*/jllama_*: {shown} -- "
            f"the export list (llama/cmake/jllama-exports.map, jllama-exports-macos.txt) or the visibility "
            f"preset is not in effect"]


def find_libraries(root):
    for dirpath, _, files in os.walk(root):
        for name in files:
            if name.endswith(LIBRARY_SUFFIXES):
                yield os.path.join(dirpath, name)


def denied(deps):
    return [d for d in deps if any(bad in d.lower() for bad in DENIED)]


def glibc_name(version, family="GLIBC"):
    return family + "_" + ".".join(str(p) for p in version)


def violations(rel, deps, alignments=(), siblings=(), runpath=None, glibc=None, glibcxx=None, cxxabi=None):
    """The violations of one library, `rel` being its path below the natives root
    (.../<OS>/<ARCH>/<backend>/<library>); `alignments` are its LOAD segment alignments (checked
    for Android libraries), `siblings` the names of the other files in its directory, `runpath`
    its ELF run path, `glibc` the highest glibc symbol version it references and `glibcxx` /
    `cxxabi` the highest libstdc++ ones."""
    failures = [f"{rel} needs {d}, which no consumer can be expected to have" for d in denied(deps)]
    parts = rel.split("/")
    key = "/".join(parts[-4:-1]) if len(parts) >= 4 else ""
    if key.startswith("Linux-Android/"):
        failures += [f"{rel}: LOAD alignment {a} is not a multiple of {ANDROID_PAGE_ALIGNMENT} "
                     f"(Google Play 16 KB page-size requirement)" for a in alignments if a % ANDROID_PAGE_ALIGNMENT]
    siblings = set(siblings)
    android = key.startswith("Linux-Android/")
    is_module = bool(key) and key.rsplit("/", 1)[-1] not in CPU_BACKENDS and not android
    if is_module and not any(d.lower() in GGML_BASE for d in deps):
        failures.append(f"{rel} does not import ggml-base, so it is not a backend module built against the shared "
                        f"ggml (a JLLAMA_MODULE_ONLY build) -- a GPU jar must hold the module alone")
    if android and rel.endswith(".so"):
        # An Android library carries NO run path, and needs none. CMake's Android platform drops the
        # flag (Modules/Platform/Android.cmake: "Android reportedly ignores RPATH, and we cannot
        # predict the install location anyway"; dockcross's toolchain file sets CMAKE_SYSTEM_NAME
        # Android, which is also why ggml builds its android_armv* variant list), so the
        # INSTALL_RPATH "$ORIGIN" of llama/CMakeLists.txt writes nothing there. And bionic resolves
        # an app's DT_NEEDED through the classloader namespace's search path -- the app's
        # native-library directory, or the APK itself with extractNativeLibs=false, where $ORIGIN
        # would be a zip path -- the same lookup System.loadLibrary("jllama") relies on;
        # test-android-emulator loads the whole modular set that way. What must not ship is a
        # build-tree path. (The first full run of the modular natives, 38065017595, had the
        # desktop rule below demand $ORIGIN here and failed 27 libraries that load on the device.)
        if runpath not in (None, ORIGIN):
            failures.append(f"{rel} carries the run path {runpath!r}: an Android library has none (bionic finds "
                            f"its siblings through the app's native-library directory), and a build-tree path "
                            f"points at the CI runner")
    elif rel.endswith(".so") and (is_module or any(d in siblings for d in deps)) and runpath != ORIGIN:
        # Two different defects reach this branch, and saying which one it is matters: the first run
        # that hit the second case (the Linux SYCL module, run 38149667195) printed the first case's
        # reason, which sent a reader looking for a missing $ORIGIN that was there all along. A wrong
        # cause costs more than a bare failure.
        entries = (runpath or "").split(":")
        needs = sorted(d for d in deps if d in siblings) or list(GGML_BASE[:1])
        if ORIGIN in entries:
            extra = [e for e in entries if e != ORIGIN]
            failures.append(f"{rel} carries the run path {runpath!r}: {ORIGIN!r} is there, but so are "
                            f"{extra} -- absolute paths of the build host, which is what ships to a consumer "
                            f"who has no such directory (or a different version of it). Every other Linux "
                            f"library carries {ORIGIN!r} and nothing else, and finds its vendor runtime through "
                            f"the consumer's own environment. Strip them in the build job rather than widening "
                            f"this rule")
        else:
            failures.append(f"{rel} needs {needs} next to it "
                            f"but its run path is {runpath!r}, which does not contain {ORIGIN!r} -- the sibling "
                            f"would be looked up on the system instead, where a foreign llama.cpp install can "
                            f"answer with a mismatched ggml ABI")
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
                        f"directory promises -- was it built in (or against) the manylinux image?")
    if ceiling:
        for family, found in (("GLIBCXX", glibcxx), ("CXXABI", cxxabi)):
            top = LIBSTDCXX_CEILING[family]
            if found and found > top:
                failures.append(f"{rel} references {glibc_name(found, family)}, above the {glibc_name(top, family)} "
                                f"of manylinux_2_28's libstdc++ -- was it linked against gcc-toolset's libstdc++ "
                                f"(the system libstdc++.so.6 plus libstdc++_nonshared.a)?")
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
    args = list(argv[1:])
    require_signed = "--require-signed" in args
    args = [a for a in args if a != "--require-signed"]
    if len(args) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    root = args[0]
    checked = 0
    failures = []
    for path in sorted(find_libraries(root)):
        checked += 1
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        siblings = {n for n in os.listdir(os.path.dirname(path)) if n != os.path.basename(path)}
        alignments, runpath, glibc, glibcxx, cxxabi, note = (), None, None, None, None, ""
        typeinfo = []
        try:
            if path.endswith(".so"):
                with open(path, "rb") as f:
                    data = f.read()
                info = elf_info(data)
                deps, runpath, glibc = info["needed"], info["runpath"], info["glibc"]
                glibcxx, cxxabi = info["glibcxx"], info["cxxabi"]
                if "/Linux-Android/" in "/" + rel:
                    alignments = elf_load_alignments(data)
                symbol_alignment = typeinfo_name_alignment(rel)
                if symbol_alignment:
                    typeinfo = typeinfo_name_violations(rel, elf_symbols(data), symbol_alignment,
                                                        required=os.path.basename(path) in JLLAMA_LIBRARIES)
                note = (f" [runpath={runpath}]" if runpath else "") + "".join(
                    f" [{glibc_name(v, family)}]" for family, v in (("GLIBC", glibc), ("GLIBCXX", glibcxx),
                                                                     ("CXXABI", cxxabi)) if v)
            else:
                deps = dependencies(path)
        except (ValueError, struct.error) as e:
            failures.append(f"{rel}: not a readable native library ({e})")
            continue
        print(f"{rel}: {' '.join(deps)}{note}")
        failures += violations(rel, deps, alignments, siblings, runpath, glibc, glibcxx, cxxabi)
        failures += typeinfo
        if os.path.basename(path) in JLLAMA_LIBRARIES:
            try:
                names = exports(path)
            except (ValueError, struct.error) as e:
                failures.append(f"{rel}: the export table could not be read ({e})")
                continue
            print(f"{rel}: {len(names)} exported symbol(s)")
            failures += export_violations(rel, names)
    if checked == 0:
        print(f"no native library found under {root}", file=sys.stderr)
        return 2
    failures += signature_violations(root, require_signed)
    for f in failures:
        print(f"::error::{f}", file=sys.stderr)
    print(f"{checked} native libraries checked, {len(failures)} violations")
    return 1 if failures else 0
