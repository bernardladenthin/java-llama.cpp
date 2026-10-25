# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

import os
import io
import contextlib
import struct
import tempfile
import unittest

from buildcheck import nativedeps


def elf64(needed, runpath=None, glibc=(), exports=(), undefined=()):
    """A minimal little-endian ELF64 with a .dynstr and a .dynamic naming `needed` (and a
    DT_RUNPATH when given), plus a .gnu.version_r requiring the `glibc` symbol versions, plus a
    .dynsym defining `exports` (global) and referencing `undefined`."""
    strtab = b"\0"
    offsets = {}
    for name in (list(needed) + ([runpath] if runpath is not None else []) + (["libc.so.6"] if glibc else [])
                 + list(glibc) + list(exports) + list(undefined)):
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
    dynsym = bytes(24)  # the null symbol
    for name in exports:  # STB_GLOBAL (1) << 4 | STT_FUNC (2), defined in section 1
        dynsym += struct.pack("<IBBHQQ", offsets[name], (1 << 4) | 2, 0, 1, 0x1000, 8)
    for name in undefined:  # SHN_UNDEF: a reference, not an export
        dynsym += struct.pack("<IBBHQQ", offsets[name], (1 << 4) | 2, 0, 0, 0, 0)
    header_size, section_size = 64, 64
    strtab_off = header_size
    dynamic_off = strtab_off + len(strtab)
    verneed_off = dynamic_off + len(dynamic)
    dynsym_off = verneed_off + len(verneed)
    shoff = dynsym_off + len(dynsym)
    header = bytearray(header_size)
    header[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<Q", header, 0x28, shoff)
    nsections = 3 + (1 if glibc else 0) + (1 if exports or undefined else 0)
    struct.pack_into("<HH", header, 0x3A, section_size, nsections)
    sections = bytes(section_size)  # SHT_NULL
    sections += struct.pack("<IIQQQQIIQQ", 0, 3, 0, 0, strtab_off, len(strtab), 0, 0, 1, 0)  # SHT_STRTAB
    sections += struct.pack("<IIQQQQIIQQ", 0, 6, 0, 0, dynamic_off, len(dynamic), 1, 0, 8, 16)  # SHT_DYNAMIC
    if glibc:
        sections += struct.pack("<IIQQQQIIQQ", 0, 0x6FFFFFFE, 0, 0, verneed_off, len(verneed), 1, 1, 4, 0)
    if exports or undefined:
        sections += struct.pack("<IIQQQQIIQQ", 0, 11, 0, 0, dynsym_off, len(dynsym), 1, 1, 8, 24)  # SHT_DYNSYM
    return bytes(header) + strtab + dynamic + verneed + dynsym + sections


def elf64_segments(alignments):
    """A minimal little-endian ELF64 with one PT_LOAD program header per alignment."""
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<Q", header, 0x20, 64)
    struct.pack_into("<HH", header, 0x36, 56, len(alignments))
    phdrs = b"".join(struct.pack("<IIQQQQQQ", 1, 5, 0, 0, 0, 0, 0, a) for a in alignments)
    return bytes(header) + phdrs


def pe64(imports, exports=(), signed=False):
    """A minimal PE32+ with one section holding an import directory naming `imports` and, when
    given, an export directory naming `exports`; `signed` adds an attribute certificate table entry
    (IMAGE_DIRECTORY_ENTRY_SECURITY) the way an Authenticode signature does."""
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
    export_rva = 0
    if exports:
        export_rva = rva + len(body)
        # IMAGE_EXPORT_DIRECTORY (40 bytes), then the name RVA table, then the names
        table_rva = export_rva + 40
        export_names = b""
        export_name_rvas = []
        for name in exports:
            export_name_rvas.append(table_rva + 4 * len(exports) + len(export_names))
            export_names += name.encode() + b"\0"
        export_dir = struct.pack("<IIHHIIIIIII", 0, 0, 0, 0, 0, 1, len(exports), len(exports), 0, table_rva, 0)
        body += export_dir + b"".join(struct.pack("<I", r) for r in export_name_rvas) + export_names
    data = bytearray(raw + len(body))
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, pe)
    struct.pack_into("<H", data, pe + 6, 1)
    struct.pack_into("<H", data, pe + 20, opt_size)
    struct.pack_into("<H", data, pe + 24, 0x20B)
    if exports:
        struct.pack_into("<II", data, pe + 24 + 112, export_rva, len(body) - (export_rva - rva))
    struct.pack_into("<I", data, pe + 24 + 112 + 8, rva)
    struct.pack_into("<IIII", data, section_table + 8, len(body), rva, len(body), raw)
    data[raw:] = body
    if signed:
        # The security directory holds a FILE OFFSET (not an RVA) of the WIN_CERTIFICATE blob,
        # appended after the sections; the check reads presence, not the blob.
        struct.pack_into("<II", data, pe + 24 + 112 + 4 * 8, len(data), 8)
        data += bytes(8)
    return bytes(data)


def macho64(dylibs, exports=(), private=()):
    """A minimal 64-bit Mach-O with LC_LOAD_DYLIB commands and, when symbols are given, an
    LC_SYMTAB whose table defines `exports` as external and `private` as non-external."""
    commands = b""
    for name in dylibs:
        payload = name.encode() + b"\0"
        size = (24 + len(payload) + 7) // 8 * 8
        commands += struct.pack("<IIIIII", 0xC, size, 24, 0, 0, 0) + payload.ljust(size - 24, b"\0")
    ncmds = len(dylibs)
    tail = b""
    if exports or private:
        ncmds += 1
        symtab_cmd_size = 24
        header_size = 32
        symoff = header_size + len(commands) + symtab_cmd_size
        strtab = b"\0"
        nlist = b""
        for name, external in [(n, True) for n in exports] + [(n, False) for n in private]:
            n_strx = len(strtab)
            strtab += name.encode() + b"\0"
            nlist += struct.pack("<IBBHQ", n_strx, 0x0E | (0x01 if external else 0), 1, 0, 0x1000)
        stroff = symoff + len(nlist)
        commands += struct.pack("<IIIIII", 0x2, symtab_cmd_size, symoff, len(exports) + len(private), stroff, len(strtab))
        tail = nlist + strtab
    header = struct.pack("<IiiIIIII", 0xFEEDFACF, 0, 0, 6, ncmds, len(commands), 0, 0)
    return header + commands + tail


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
        self.assertEqual(info, {"needed": ["libggml.so", "libc.so.6"], "runpath": "$ORIGIN", "glibc": (2, 28),
                                "glibcxx": None, "cxxabi": None})

    def test_elf_libstdcxx_versions(self):
        # the highest of each family; s390x's long double twins count as their family, and the
        # versions without a number carry no floor (a crash on int("PRIVATE") before)
        info = nativedeps.elf_info(elf64(["libstdc++.so.6", "libc.so.6"], glibc=[
            "GLIBC_2.17", "GLIBC_PRIVATE", "GLIBCXX_3.4.9", "GLIBCXX_3.4.21", "GLIBCXX_LDBL_3.4.25",
            "CXXABI_1.3.9", "CXXABI_TM_1", "CXXABI_LDBL_1.3"]))
        self.assertEqual((info["glibc"], info["glibcxx"], info["cxxabi"]), ((2, 17), (3, 4, 25), (1, 3, 9)))

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


class ExportsTest(unittest.TestCase):
    """The export set of a jllama library: JNI entry points and the jllama_* C API, nothing else."""

    def test_elf_exports_are_the_defined_global_symbols_of_dynsym(self):
        data = elf64(["libc.so.6"], exports=["JNI_OnLoad", "Java_x", "llama_model_load"], undefined=["dlopen"])
        self.assertEqual(nativedeps.elf_exports(data), ["JNI_OnLoad", "Java_x", "llama_model_load"])
        self.assertEqual(nativedeps.elf_exports(elf64(["libc.so.6"])), [])

    def test_pe_exports_are_the_export_directory_names(self):
        data = pe64(["KERNEL32.dll"], exports=["JNI_OnLoad", "Java_x"])
        self.assertEqual(nativedeps.pe_exports(data), ["JNI_OnLoad", "Java_x"])
        self.assertEqual(nativedeps.pe_imports(data), ["KERNEL32.dll"], "the import directory still reads")
        self.assertEqual(nativedeps.pe_exports(pe64(["KERNEL32.dll"])), [])

    def test_macho_exports_are_the_external_defined_symbols(self):
        data = macho64(["/usr/lib/libSystem.B.dylib"], exports=["_JNI_OnLoad", "_Java_x"], private=["_helper"])
        self.assertEqual(nativedeps.macho_exports(data), ["_JNI_OnLoad", "_Java_x"])
        self.assertEqual(nativedeps.macho_dylibs(data), ["/usr/lib/libSystem.B.dylib"])

    def test_the_allowed_export_set_passes_and_a_stray_symbol_is_named(self):
        self.assertEqual(nativedeps.export_violations("x/libjllama.so",
                                                      ["JNI_OnLoad", "JNI_OnUnload", "Java_net_ladenthin_llama_LlamaModel_loadModel",
                                                       "jllama_api_version"]), [])
        found = nativedeps.export_violations("x/libjllama.so", ["JNI_OnLoad", "Java_a", "llama_model_load", "ggml_init"])
        self.assertEqual(len(found), 1)
        self.assertIn("exports 2 symbol(s) outside", found[0])
        self.assertIn("llama_model_load, ggml_init", found[0])
        self.assertIn("jllama-exports.map", found[0])

    def test_a_library_without_a_jni_entry_point_is_never_a_pass(self):
        found = nativedeps.export_violations("x/libjllama.so", [])
        self.assertEqual(len(found), 1)
        self.assertIn("no Java_* entry point", found[0])
        self.assertTrue(nativedeps.export_violations("x/libjllama.so", ["jllama_api_version"]))

    def test_main_checks_the_jllama_library_and_leaves_the_others_alone(self):
        with tempfile.TemporaryDirectory() as root:
            d = os.path.join(root, "Linux", "x86_64", "cpu")
            os.makedirs(d)
            with open(os.path.join(d, "libjllama.so"), "wb") as f:
                f.write(elf64(["libc.so.6", "libstdc++.so.6", "libm.so.6", "libgcc_s.so.1", "libdl.so.2",
                               "libpthread.so.0", "librt.so.1", "ld-linux-x86-64.so.2"],
                              runpath="$ORIGIN", glibc=["GLIBC_2.28"], exports=["JNI_OnLoad", "Java_a", "ggml_init"]))
            with open(os.path.join(d, "libggml-base.so"), "wb") as f:
                f.write(elf64(["libc.so.6", "libstdc++.so.6", "libm.so.6", "libgcc_s.so.1"],
                              runpath="$ORIGIN", glibc=["GLIBC_2.28"], exports=["ggml_init"]))
            out = io.StringIO()
            err = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = nativedeps.main(["x", root])
            self.assertEqual(rc, 1)
            self.assertIn("libjllama.so: exports 1 symbol(s) outside", err.getvalue())
            self.assertNotIn("libggml-base.so: exports", err.getvalue())
            self.assertIn("libjllama.so: 3 exported symbol(s)", out.getvalue())


class SignatureTest(unittest.TestCase):

    def _tree(self, root, signed_state, dll_signed):
        d = os.path.join(root, "net/ladenthin/llama/Windows/x86_64/cpu")
        os.makedirs(d)
        with open(os.path.join(d, "jllama.dll"), "wb") as f:
            f.write(pe64(["kernel32.dll"], exports=["JNI_OnLoad", "Java_a"], signed=dll_signed))
        with open(os.path.join(d, "ggml.dll"), "wb") as f:
            f.write(pe64(["kernel32.dll"], exports=["ggml_init"], signed=dll_signed))
        if signed_state is not None:
            with open(os.path.join(d, "jllama-build.txt"), "w", encoding="utf-8") as f:
                f.write(f"llama.cpp=b1\nsigned={signed_state}\n")
        return root

    def test_a_signed_pe_is_recognised_and_an_unsigned_one_is_not(self):
        self.assertTrue(nativedeps.pe_signed(pe64(["kernel32.dll"], signed=True)))
        self.assertFalse(nativedeps.pe_signed(pe64(["kernel32.dll"])))

    def test_signed_yes_over_an_unsigned_dll_names_every_such_file(self):
        with tempfile.TemporaryDirectory() as root:
            failures = nativedeps.signature_violations(self._tree(root, "yes", False))
            self.assertEqual(len(failures), 2)
            self.assertIn("Windows/x86_64/cpu/jllama.dll: jllama-build.txt says signed=yes", failures[1])

    def test_signed_yes_over_signed_dlls_passes_and_signed_no_is_fine_unless_required(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(nativedeps.signature_violations(self._tree(root, "yes", True)), [])
        with tempfile.TemporaryDirectory() as root:
            self._tree(root, "no", False)
            self.assertEqual(nativedeps.signature_violations(root), [])
            failures = nativedeps.signature_violations(root, require_signed=True)
            self.assertEqual(len(failures), 2)
            self.assertIn("unsigned (jllama-build.txt says signed=no) while a signature is required", failures[0])

    def test_a_directory_without_a_build_file_is_unsigned_when_required(self):
        with tempfile.TemporaryDirectory() as root:
            self._tree(root, None, False)
            self.assertEqual(nativedeps.signature_violations(root), [])
            self.assertEqual(len(nativedeps.signature_violations(root, require_signed=True)), 2)

    def test_non_windows_directories_are_never_subject_to_the_signature_rule(self):
        with tempfile.TemporaryDirectory() as root:
            d = os.path.join(root, "net/ladenthin/llama/Linux/x86_64/cpu")
            os.makedirs(d)
            with open(os.path.join(d, "libjllama.so"), "wb") as f:
                f.write(elf64(["libc.so.6"], exports=["JNI_OnLoad", "Java_a"]))
            self.assertEqual(nativedeps.signature_violations(root, require_signed=True), [])

    def test_main_takes_the_flag_before_the_root(self):
        with tempfile.TemporaryDirectory() as root:
            self._tree(root, "no", False)
            with open(os.path.join(root, "net/ladenthin/llama/Windows/x86_64/cpu/jllama.dll"), "wb") as f:
                f.write(pe64(["kernel32.dll", "api-ms-win-crt-runtime-l1-1-0.dll"] , exports=["JNI_OnLoad", "Java_a"]))
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = nativedeps.main(["x", "--require-signed", root])
            self.assertEqual(rc, 1)
            self.assertIn("while a signature is required", err.getvalue())
            self.assertEqual(nativedeps.main(["x", "--require-signed"]), 2)


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
                                               ["libggml-base.so", "libc.so.6"], runpath="$ORIGIN", glibc=(2, 39),
                                               glibcxx=(3, 4, 33), cxxabi=(1, 3, 15)), [])

    def test_s390x_keeps_the_manylinux_floor(self):
        # what the 6.0.0 jar, cross-built against the ubuntu runner's sysroot, referenced
        rel = "x/Linux/s390x/cpu/libjllama.so"
        deps = ["libstdc++.so.6", "libm.so.6", "libgcc_s.so.1", "libc.so.6", "ld64.so.1"]
        failures = nativedeps.violations(rel, deps, glibc=(2, 38), glibcxx=(3, 4, 32), cxxabi=(1, 3, 13))
        self.assertEqual(len(failures), 3, failures)
        for found, floor in (("GLIBC_2.38", "GLIBC_2.28"), ("GLIBCXX_3.4.32", "GLIBCXX_3.4.25"),
                             ("CXXABI_1.3.13", "CXXABI_1.3.11")):
            self.assertTrue(any(found in f and floor in f for f in failures), (found, failures))
        # the build against the manylinux_2_28 sysroot: glibc 2.28 names libpthread and libdl itself
        self.assertEqual(nativedeps.violations(rel, deps + ["libpthread.so.0", "libdl.so.2"], glibc=(2, 27),
                                               glibcxx=(3, 4, 25), cxxabi=(1, 3, 11)), [])

    def test_the_manylinux_directories_keep_their_libstdcxx_floor(self):
        for rel in ("x/Linux/x86_64/cpu/libllama.so", "x/Linux/aarch64/cpu/libllama.so"):
            self.assertEqual(nativedeps.violations(rel, ["libc.so.6"], glibcxx=(3, 4, 25), cxxabi=(1, 3, 11)), [])
            failures = nativedeps.violations(rel, ["libc.so.6"], glibcxx=(3, 4, 26))
            self.assertEqual(len(failures), 1, failures)
            self.assertIn("GLIBCXX_3.4.26", failures[0])
            self.assertIn("libstdc++_nonshared.a", failures[0])
        rel = "x/Linux/x86_64/cuda13/libggml-cuda.so"
        self.assertEqual(len(nativedeps.violations(rel, ["libggml-base.so", "libc.so.6"], runpath="$ORIGIN",
                                                   cxxabi=(1, 3, 12))), 1)

    def test_gpu_modules_are_held_to_the_denylist_and_must_be_modules(self):
        rel = "x/Linux/x86_64/cuda13/libggml-cuda.so"
        module = ["libggml-base.so", "libcudart.so.13", "libc.so.6"]
        self.assertEqual(nativedeps.violations(rel, module, runpath="$ORIGIN"), [])
        self.assertIn("no consumer can be expected",
                      nativedeps.violations(rel, module + ["libibverbs.so.1"], runpath="$ORIGIN")[0])
        # a monolithic library in a module directory (the layout before 6.0.0) is refused: it does not
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


class OpenMpDependencyTest(unittest.TestCase):
    """The Linux jars must not regain libgomp, and this is the check that keeps it that way.

    llama/CMakeLists.txt forces GGML_OPENMP=OFF on Linux because libggml-base.so otherwise declares
    libgomp.so.1 -- which a plain Ubuntu server image and a distroless Java image do not have, so the
    whole jar fails to load with a message that reads like a packaging defect. Removing libgomp from
    ALLOWED is what turns a regression into a red `package` job instead of an unloadable release.

    The failure mode this guards against is social rather than technical: when a future build regains
    the dependency, the quickest way to make CI green again is to put libgomp back in the allowlist.
    These assertions say in the test suite that doing so defeats the purpose.
    """

    def test_the_linux_cpu_allowlists_do_not_permit_libgomp(self):
        for directory in ("Linux/x86_64/cpu", "Linux/aarch64/cpu", "Linux/s390x/cpu"):
            self.assertIn(directory, nativedeps.ALLOWED, directory)
            self.assertNotIn(
                "libgomp.so.1",
                nativedeps.ALLOWED[directory],
                directory + ": GGML_OPENMP=OFF is forced for this platform, so a build that needs "
                "libgomp is a regression -- fix the build, do not widen the allowlist",
            )

    def test_the_runtimes_the_jvm_itself_needs_stay_allowed(self):
        # The counterpart, so the entry above is not read as "strip every dependency": libstdc++ and
        # libgcc_s stay dynamic on purpose, because every image that can run this library has them --
        # HotSpot's own libjvm.so links both -- and linking them statically would give each of our
        # shared objects its own unwinder.
        for directory in ("Linux/x86_64/cpu", "Linux/aarch64/cpu"):
            self.assertIn("libstdc++.so.6", nativedeps.ALLOWED[directory], directory)
            self.assertIn("libgcc_s.so.1", nativedeps.ALLOWED[directory], directory)


class MainTest(unittest.TestCase):

    def test_checks_a_tree_and_reports_violations(self):
        with tempfile.TemporaryDirectory() as root:
            ok = os.path.join(root, "net/ladenthin/llama/Linux/s390x/cpu")
            bad = os.path.join(root, "net/ladenthin/llama/Windows/aarch64/cpu")
            os.makedirs(ok)
            os.makedirs(bad)
            with open(os.path.join(ok, "libjllama.so"), "wb") as f:
                f.write(elf64(["libc.so.6"], exports=["JNI_OnLoad", "Java_a"]))
            self.assertEqual(nativedeps.main(["x", root]), 0)
            with open(os.path.join(bad, "jllama.dll"), "wb") as f:
                f.write(pe64(["kernel32.dll", "libomp140.x86_64.dll"], exports=["JNI_OnLoad", "Java_a"]))
            self.assertEqual(nativedeps.main(["x", root]), 1)

    def test_checks_every_library_of_a_variant_directory(self):
        with tempfile.TemporaryDirectory() as root:
            cpu = os.path.join(root, "net/ladenthin/llama/Linux/x86_64/cpu")
            os.makedirs(cpu)
            with open(os.path.join(cpu, "libjllama.so"), "wb") as f:
                f.write(elf64(["libggml.so", "libc.so.6"], runpath="$ORIGIN", glibc=["GLIBC_2.28"],
                              exports=["JNI_OnLoad", "Java_a"]))
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


class RunPathDiagnosticTest(unittest.TestCase):
    """The two ways a module's run path can be wrong say different things.

    Both still fail -- the rule is unchanged. What is new is that the message names the actual
    defect: the Linux SYCL module (run 38149667195) carried `$ORIGIN` *plus* oneAPI's install paths
    and was reported as if `$ORIGIN` were missing, which is a wrong cause rather than a vague one.
    """

    def _fail(self, runpath):
        return nativedeps.violations(
            "net/ladenthin/llama/Linux/x86_64/sycl/libggml-sycl.so",
            ["libggml-base.so"],
            siblings=("libggml-base.so",),
            runpath=runpath,
        )

    def test_origin_alone_is_the_only_accepted_run_path(self):
        self.assertEqual([], self._fail("$ORIGIN"))

    def test_origin_plus_build_host_paths_names_the_extra_entries(self):
        failures = self._fail("$ORIGIN:/opt/intel/oneapi/mkl/2026.1/lib")
        self.assertEqual(1, len(failures))
        self.assertIn("/opt/intel/oneapi/mkl/2026.1/lib", failures[0])
        self.assertIn("build host", failures[0])
        # and it must NOT claim the sibling is unreachable, which is the other defect
        self.assertNotIn("looked up on the system", failures[0])

    def test_a_run_path_without_origin_says_the_sibling_is_unreachable(self):
        failures = self._fail("/usr/lib/x86_64-linux-gnu")
        self.assertEqual(1, len(failures))
        self.assertIn("looked up on the system", failures[0])
        self.assertIn("mismatched ggml ABI", failures[0])

    def test_a_missing_run_path_is_still_a_failure(self):
        self.assertEqual(1, len(self._fail(None)))
