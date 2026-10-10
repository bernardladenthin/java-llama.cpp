# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

import os
import struct
import tempfile
import unittest

from buildcheck import nativedeps


def elf64(needed, runpath=None, glibc=()):
    """A minimal little-endian ELF64 with a .dynstr and a .dynamic naming `needed` (and a
    DT_RUNPATH when given), plus a .gnu.version_r requiring the `glibc` symbol versions."""
    strtab = b"\0"
    offsets = {}
    for name in list(needed) + ([runpath] if runpath is not None else []) + (["libc.so.6"] if glibc else []) + list(glibc):
        offsets[name] = len(strtab)
        strtab += name.encode() + b"\0"
    dynamic = b"".join(struct.pack("<qQ", 1, offsets[n]) for n in needed)
    if runpath is not None:
        dynamic += struct.pack("<qQ", 29, offsets[runpath])
    dynamic += struct.pack("<qQ", 0, 0)
    verneed = b""
    if glibc:
        verneed = struct.pack("<HHIII", 1, len(glibc), offsets["libc.so.6"], 16, 0)
        for i, v in enumerate(glibc):
            verneed += struct.pack("<IHHII", 0, 0, 2 + i, offsets[v], 16 if i + 1 < len(glibc) else 0)
    header_size, section_size = 64, 64
    strtab_off = header_size
    dynamic_off = strtab_off + len(strtab)
    verneed_off = dynamic_off + len(dynamic)
    shoff = verneed_off + len(verneed)
    header = bytearray(header_size)
    header[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<Q", header, 0x28, shoff)
    struct.pack_into("<HH", header, 0x3A, section_size, 4 if glibc else 3)
    sections = bytes(section_size)  # SHT_NULL
    sections += struct.pack("<IIQQQQIIQQ", 0, 3, 0, 0, strtab_off, len(strtab), 0, 0, 1, 0)  # SHT_STRTAB
    sections += struct.pack("<IIQQQQIIQQ", 0, 6, 0, 0, dynamic_off, len(dynamic), 1, 0, 8, 16)  # SHT_DYNAMIC
    if glibc:
        sections += struct.pack("<IIQQQQIIQQ", 0, 0x6FFFFFFE, 0, 0, verneed_off, len(verneed), 1, 1, 4, 0)
    return bytes(header) + strtab + dynamic + verneed + sections


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

    def test_elf_run_path_and_glibc_versions(self):
        plain = nativedeps.elf_info(elf64(["libc.so.6"]))
        self.assertEqual((plain["runpath"], plain["glibc"]), (None, None))
        info = nativedeps.elf_info(elf64(["libggml.so", "libc.so.6"], runpath="$ORIGIN",
                                         glibc=["GLIBC_2.28", "GLIBC_2.2.5", "GLIBC_2.17"]))
        self.assertEqual(info, {"needed": ["libggml.so", "libc.so.6"], "runpath": "$ORIGIN", "glibc": (2, 28)})

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
        # The UCRT forwarder is not incidental: a Windows CPU library must import one (see
        # test_a_windows_cpu_library_without_the_ucrt_forwarders_is_a_violation). This case is
        # about the allowlist matching regardless of case.
        self.assertEqual(nativedeps.violations("x/Windows/x86_64/cpu/jllama.dll",
                                               ["KERNEL32.dll", "WS2_32.dll",
                                                "API-MS-WIN-CRT-RUNTIME-L1-1-0.DLL"]), [])

    def test_a_new_cpu_dependency_fails(self):
        failures = nativedeps.violations("x/Linux/x86_64/cpu/libjllama.so", ["libc.so.6", "libssl.so.3"])
        self.assertEqual(len(failures), 1)
        self.assertIn("libssl.so.3", failures[0])

    def test_a_cpu_directory_without_allowlist_fails(self):
        self.assertIn("no dependency allowlist", nativedeps.violations("x/Linux/riscv64/cpu/libjllama.so", [])[0])

    def test_a_sibling_is_allowed_when_found_through_origin(self):
        rel = "x/Linux/x86_64/cpu/libjllama.so"
        siblings = {"libggml.so", "libggml-base.so", "jllama-files.txt"}
        self.assertEqual(nativedeps.violations(rel, ["libggml.so", "libc.so.6"], siblings=siblings, runpath="$ORIGIN"), [])
        # the same dependency without the sibling next to it is a new runtime dependency
        self.assertIn("did not need before", nativedeps.violations(rel, ["libggml.so", "libc.so.6"], runpath="$ORIGIN")[0])
        # a sibling the dynamic linker would look up on the system instead
        for runpath in (None, "/home/runner/work/llama/build:$ORIGIN"):
            failures = nativedeps.violations(rel, ["libggml.so"], siblings=siblings, runpath=runpath)
            self.assertEqual(len(failures), 1, failures)
            self.assertIn("$ORIGIN", failures[0])
        # a module needing nothing from its directory needs no run path
        self.assertEqual(nativedeps.violations("x/Linux/x86_64/cpu/libggml-base.so", ["libc.so.6"], siblings=siblings), [])

    def test_the_manylinux_directories_keep_their_glibc_floor(self):
        rel = "x/Linux/aarch64/cpu/libggml-cpu-armv9.2_1.so"
        self.assertEqual(nativedeps.violations(rel, ["libc.so.6"], glibc=(2, 28)), [])
        failures = nativedeps.violations(rel, ["libc.so.6"], glibc=(2, 29))
        self.assertEqual(len(failures), 1, failures)
        self.assertIn("GLIBC_2.29", failures[0])
        self.assertIn("GLIBC_2.28", failures[0])
        # a build on the ubuntu runner promises no floor
        self.assertEqual(nativedeps.violations("x/Linux/x86_64/vulkan/libggml-vulkan.so",
                                               ["libggml-base.so", "libc.so.6"], runpath="$ORIGIN", glibc=(2, 39)), [])

    def test_gpu_modules_are_held_to_the_denylist_and_must_be_modules(self):
        rel = "x/Linux/x86_64/cuda13/libggml-cuda.so"
        module = ["libggml-base.so", "libcudart.so.13", "libc.so.6"]
        self.assertEqual(nativedeps.violations(rel, module, runpath="$ORIGIN"), [])
        self.assertIn("no consumer can be expected",
                      nativedeps.violations(rel, module + ["libibverbs.so.1"], runpath="$ORIGIN")[0])
        # a monolithic library in a module directory (the layout before 5.2.0) is refused: it does not
        # import ggml-base, because it carries its own
        found = nativedeps.violations("x/Linux/x86_64/cuda13/libjllama.so", ["libcudart.so.13", "libc.so.6"],
                                      runpath="$ORIGIN")
        self.assertEqual(len(found), 1, found)
        self.assertIn("does not import ggml-base", found[0])
        # the module finds libggml-base next to it only through $ORIGIN
        found = nativedeps.violations(rel, module, runpath=None)
        self.assertEqual(len(found), 1, found)
        self.assertIn("$ORIGIN", found[0])
        # Windows: ggml-base.dll by name, no run path to check
        self.assertEqual(nativedeps.violations("x/Windows/x86_64/vulkan/ggml-vulkan.dll",
                                               ["ggml-base.dll", "vulkan-1.dll", "KERNEL32.dll"]), [])
        self.assertIn("does not import ggml-base",
                      nativedeps.violations("x/Windows/x86_64/vulkan/jllama.dll", ["vulkan-1.dll"])[0])


    def test_the_android_opencl_module_is_held_to_bionic_plus_the_icd_and_ggml_base(self):
        rel = "x/Linux-Android/aarch64/opencl/libggml-opencl.so"
        self.assertEqual(nativedeps.violations(rel, ["libc.so", "libOpenCL.so", "liblog.so", "libggml-base.so"]), [])
        self.assertIn("libomp.so", nativedeps.violations(rel, ["libc.so", "libomp.so", "libggml-base.so"])[0])
        self.assertIn("libOpenCL.so", nativedeps.violations("x/Linux-Android/aarch64/cpu/libjllama.so",
                                                            ["libOpenCL.so"])[0])

    def test_android_libraries_need_16_kb_aligned_segments(self):
        rel = "x/Linux-Android/x86_64/cpu/libjllama.so"
        self.assertEqual(nativedeps.violations(rel, ["libc.so"], [16384, 65536]), [])
        self.assertIn("LOAD alignment 4096", nativedeps.violations(rel, ["libc.so"], [16384, 4096])[0])
        # a desktop library is not held to it
        self.assertEqual(nativedeps.violations("x/Linux/s390x/cpu/libjllama.so", ["libc.so.6"], [4096]), [])

    def test_android_libraries_carry_no_run_path(self):
        # bionic finds the siblings through the app's native-library directory (the lookup
        # System.loadLibrary itself uses) and CMake's Android platform emits no run path, so none is
        # demanded; only a build-tree path fails. Run 38065017595 failed 27 libraries that load on the
        # device because the desktop rule was applied here.
        rel = "x/Linux-Android/aarch64/cpu/libggml-cpu-android_armv8.2_1.so"
        siblings = {"libjllama.so", "libggml.so", "libggml-base.so", "jllama-files.txt"}
        deps = ["libggml-base.so", "libm.so", "libdl.so", "libc.so"]
        self.assertEqual(nativedeps.violations(rel, deps, siblings=siblings), [])
        self.assertEqual(nativedeps.violations(rel, deps, siblings=siblings, runpath="$ORIGIN"), [])
        failures = nativedeps.violations(rel, deps, siblings=siblings, runpath="/home/runner/work/llama/build")
        self.assertEqual(len(failures), 1, failures)
        self.assertIn("/home/runner/work/llama/build", failures[0])
        self.assertIn("build-tree path", failures[0])
        # the OpenCL module finds the CPU AAR's libggml-base the same way
        self.assertEqual(nativedeps.violations("x/Linux-Android/aarch64/opencl/libggml-opencl.so",
                                               ["libggml-base.so", "libOpenCL.so", "libm.so", "libdl.so", "libc.so"],
                                               siblings={"jllama-build.txt", "jllama-files.txt"}), [])


class MainTest(unittest.TestCase):

    def test_checks_a_tree_and_reports_violations(self):
        with tempfile.TemporaryDirectory() as root:
            ok = os.path.join(root, "net/ladenthin/llama/Linux/s390x/cpu")
            bad = os.path.join(root, "net/ladenthin/llama/Windows/aarch64/cpu")
            os.makedirs(ok)
            os.makedirs(bad)
            with open(os.path.join(ok, "libjllama.so"), "wb") as f:
                f.write(elf64(["libc.so.6"]))
            self.assertEqual(nativedeps.main(["x", root]), 0)
            with open(os.path.join(bad, "jllama.dll"), "wb") as f:
                f.write(pe64(["kernel32.dll", "libomp140.x86_64.dll"]))
            self.assertEqual(nativedeps.main(["x", root]), 1)

    def test_checks_every_library_of_a_variant_directory(self):
        with tempfile.TemporaryDirectory() as root:
            cpu = os.path.join(root, "net/ladenthin/llama/Linux/x86_64/cpu")
            os.makedirs(cpu)
            with open(os.path.join(cpu, "libjllama.so"), "wb") as f:
                f.write(elf64(["libggml.so", "libc.so.6"], runpath="$ORIGIN", glibc=["GLIBC_2.28"]))
            with open(os.path.join(cpu, "libggml.so"), "wb") as f:
                f.write(elf64(["libggml-base.so", "libc.so.6"], runpath="$ORIGIN", glibc=["GLIBC_2.17"]))
            with open(os.path.join(cpu, "libggml-base.so"), "wb") as f:
                f.write(elf64(["libc.so.6"], glibc=["GLIBC_2.27"]))
            with open(os.path.join(cpu, "jllama-files.txt"), "w", encoding="utf-8", newline="") as f:
                f.write("libggml.so\nlibggml-base.so\n")
            self.assertEqual(nativedeps.main(["x", root]), 0)
            # a module, not only the main library, is held to the floor
            with open(os.path.join(cpu, "libggml-base.so"), "wb") as f:
                f.write(elf64(["libc.so.6"], glibc=["GLIBC_2.34"]))
            self.assertEqual(nativedeps.main(["x", root]), 1)

    def test_a_windows_cpu_library_without_the_ucrt_forwarders_is_a_violation(self):
        """The other direction of the allowlist: a static UCRT only REMOVES imports, so only a
        positive requirement catches a build that lost the hybrid-CRT linker flags."""
        hybrid = ["KERNEL32.dll", "api-ms-win-crt-runtime-l1-1-0.dll"]
        self.assertEqual(nativedeps.violations("x/Windows/x86_64/cpu/jllama.dll", hybrid), [])
        static_ucrt = ["KERNEL32.dll"]
        found = nativedeps.violations("x/Windows/x86_64/cpu/jllama.dll", static_ucrt)
        self.assertEqual(len(found), 1, found)
        self.assertIn("statically linked", found[0])
        # the modules of a variants build are held to it as well, not only the main library
        self.assertEqual(len(nativedeps.violations("x/Windows/x86_64/cpu/ggml-cpu-zen4.dll", static_ucrt)), 1)
        # a GPU module directory is on a denylist and keeps its vendor runtime free
        self.assertEqual(nativedeps.violations("x/Windows/x86_64/cuda13/ggml-cuda.dll",
                                               static_ucrt + ["ggml-base.dll"]), [])
        # and a non-Windows directory is unaffected
        self.assertEqual(nativedeps.violations("x/Mac/aarch64/metal/libjllama.dylib",
                                               ["/usr/lib/libSystem.B.dylib"]), [])

    def test_any_ucrt_forwarder_is_allowed_not_an_enumerated_eleven(self):
        # Which forwarders a library imports depends on the CRT functions it happens to use, so it
        # differs between architectures and compilers. Only "the UCRT is dynamic" is the invariant.
        for extra in ("api-ms-win-crt-process-l1-1-0.dll",   # outside the set measured on one build
                      "api-ms-win-crt-conio-l1-1-0.dll",
                      "API-MS-WIN-CRT-PRIVATE-L1-1-0.DLL"):
            self.assertEqual(nativedeps.violations("x/Windows/aarch64/cpu/jllama.dll",
                                                   ["KERNEL32.dll", extra]), [], extra)
        # it is a prefix rule, not "anything that looks like an api set"
        found = nativedeps.violations("x/Windows/aarch64/cpu/jllama.dll",
                                      ["KERNEL32.dll", "api-ms-win-crt-runtime-l1-1-0.dll",
                                       "api-ms-win-core-synch-l1-2-0.dll"])
        self.assertEqual(len(found), 1, found)
        self.assertIn("api-ms-win-core-synch", found[0])

    def test_nothing_to_check_is_a_failure(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(nativedeps.main(["x", root]), 2)
        self.assertEqual(nativedeps.main(["x"]), 2)


if __name__ == "__main__":
    unittest.main()
