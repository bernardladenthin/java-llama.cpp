# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT

import os
import struct
import tempfile
import unittest

from buildcheck import nativedeps


def elf64(needed):
    """A minimal little-endian ELF64 with a .dynstr and a .dynamic naming `needed`."""
    strtab = b"\0"
    offsets = []
    for name in needed:
        offsets.append(len(strtab))
        strtab += name.encode() + b"\0"
    dynamic = b"".join(struct.pack("<qQ", 1, o) for o in offsets) + struct.pack("<qQ", 0, 0)
    header_size, section_size = 64, 64
    strtab_off = header_size
    dynamic_off = strtab_off + len(strtab)
    shoff = dynamic_off + len(dynamic)
    header = bytearray(header_size)
    header[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<Q", header, 0x28, shoff)
    struct.pack_into("<HH", header, 0x3A, section_size, 3)
    sections = bytes(section_size)  # SHT_NULL
    sections += struct.pack("<IIQQQQIIQQ", 0, 3, 0, 0, strtab_off, len(strtab), 0, 0, 1, 0)  # SHT_STRTAB
    sections += struct.pack("<IIQQQQIIQQ", 0, 6, 0, 0, dynamic_off, len(dynamic), 1, 0, 8, 16)  # SHT_DYNAMIC
    return bytes(header) + strtab + dynamic + sections


def elf64_segments(alignments):
    """A minimal little-endian ELF64 with one PT_LOAD program header per alignment."""
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<Q", header, 0x20, 64)
    struct.pack_into("<HH", header, 0x36, 56, len(alignments))
    phdrs = b"".join(struct.pack("<IIQQQQQQ", 1, 5, 0, 0, 0, 0, 0, a) for a in alignments)
    return bytes(header) + phdrs


def pe64(imports):
    """A minimal PE32+ with one section holding an import directory naming `imports`."""
    pe = 0x40
    opt_size = 240
    section_table = pe + 24 + opt_size
    raw = 0x200
    rva = 0x1000
    names = b""
    name_rvas = []
    directory_size = 20 * (len(imports) + 1)
    for name in imports:
        name_rvas.append(rva + directory_size + len(names))
        names += name.encode() + b"\0"
    directory = b"".join(struct.pack("<IIIII", 0, 0, 0, n, 0) for n in name_rvas) + bytes(20)
    body = directory + names
    data = bytearray(raw + len(body))
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, pe)
    struct.pack_into("<H", data, pe + 6, 1)
    struct.pack_into("<H", data, pe + 20, opt_size)
    struct.pack_into("<H", data, pe + 24, 0x20B)
    struct.pack_into("<I", data, pe + 24 + 112 + 8, rva)
    struct.pack_into("<IIII", data, section_table + 8, len(body), rva, len(body), raw)
    data[raw:] = body
    return bytes(data)


def macho64(dylibs):
    commands = b""
    for name in dylibs:
        payload = name.encode() + b"\0"
        size = (24 + len(payload) + 7) // 8 * 8
        commands += struct.pack("<IIIIII", 0xC, size, 24, 0, 0, 0) + payload.ljust(size - 24, b"\0")
    header = struct.pack("<IiiIIIII", 0xFEEDFACF, 0, 0, 6, len(dylibs), len(commands), 0, 0)
    return header + commands


class ParserTest(unittest.TestCase):

    def test_elf(self):
        self.assertEqual(nativedeps.elf_needed(elf64(["libc.so.6", "libm.so.6"])), ["libc.so.6", "libm.so.6"])
        with self.assertRaises(ValueError):
            nativedeps.elf_needed(b"MZ")

    def test_elf_load_alignments(self):
        self.assertEqual(nativedeps.elf_load_alignments(elf64_segments([16384, 65536])), [16384, 65536])
        with self.assertRaises(ValueError):
            nativedeps.elf_load_alignments(b"MZ")

    def test_pe(self):
        self.assertEqual(nativedeps.pe_imports(pe64(["KERNEL32.dll", "WS2_32.dll"])), ["KERNEL32.dll", "WS2_32.dll"])

    def test_macho(self):
        self.assertEqual(nativedeps.macho_dylibs(macho64(["/usr/lib/libSystem.B.dylib"])), ["/usr/lib/libSystem.B.dylib"])
        with self.assertRaises(ValueError):
            nativedeps.macho_dylibs(struct.pack("<I", 0xFEEDFACE))


class ViolationsTest(unittest.TestCase):

    def test_allowed_cpu_dependencies_pass_case_insensitively(self):
        self.assertEqual(nativedeps.violations("x/Windows/x86_64/cpu/jllama.dll", ["KERNEL32.dll", "WS2_32.dll"]), [])

    def test_a_new_cpu_dependency_fails(self):
        failures = nativedeps.violations("x/Linux/x86_64/cpu/libjllama.so", ["libc.so.6", "libssl.so.3"])
        self.assertEqual(len(failures), 1)
        self.assertIn("libssl.so.3", failures[0])

    def test_a_cpu_directory_without_allowlist_fails(self):
        self.assertIn("no dependency allowlist", nativedeps.violations("x/Linux/riscv64/cpu/libjllama.so", [])[0])

    def test_gpu_backends_are_held_to_the_denylist_only(self):
        rel = "x/Linux/x86_64/cuda13/libjllama.so"
        self.assertEqual(nativedeps.violations(rel, ["libcudart.so.13"]), [])
        self.assertIn("no consumer can be expected", nativedeps.violations(rel, ["libibverbs.so.1"])[0])


    def test_the_android_opencl_build_is_held_to_bionic_plus_the_icd(self):
        rel = "x/Linux-Android/aarch64/opencl/libjllama.so"
        self.assertEqual(nativedeps.violations(rel, ["libc.so", "libOpenCL.so", "liblog.so"]), [])
        self.assertIn("libomp.so", nativedeps.violations(rel, ["libc.so", "libomp.so"])[0])
        self.assertIn("libOpenCL.so", nativedeps.violations("x/Linux-Android/aarch64/cpu/libjllama.so",
                                                            ["libOpenCL.so"])[0])

    def test_android_libraries_need_16_kb_aligned_segments(self):
        rel = "x/Linux-Android/x86_64/cpu/libjllama.so"
        self.assertEqual(nativedeps.violations(rel, ["libc.so"], [16384, 65536]), [])
        self.assertIn("LOAD alignment 4096", nativedeps.violations(rel, ["libc.so"], [16384, 4096])[0])
        # a desktop library is not held to it
        self.assertEqual(nativedeps.violations("x/Linux/s390x/cpu/libjllama.so", ["libc.so.6"], [4096]), [])


class MainTest(unittest.TestCase):

    def test_checks_a_tree_and_reports_violations(self):
        with tempfile.TemporaryDirectory() as root:
            ok = os.path.join(root, "net/ladenthin/llama/Linux/s390x/cpu")
            bad = os.path.join(root, "net/ladenthin/llama/Windows/x86/cpu")
            os.makedirs(ok)
            os.makedirs(bad)
            with open(os.path.join(ok, "libjllama.so"), "wb") as f:
                f.write(elf64(["libc.so.6"]))
            self.assertEqual(nativedeps.main(["x", root]), 0)
            with open(os.path.join(bad, "jllama.dll"), "wb") as f:
                f.write(pe64(["kernel32.dll", "libomp140.x86_64.dll"]))
            self.assertEqual(nativedeps.main(["x", root]), 1)

    def test_nothing_to_check_is_a_failure(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(nativedeps.main(["x", root]), 2)
        self.assertEqual(nativedeps.main(["x"]), 2)


if __name__ == "__main__":
    unittest.main()
