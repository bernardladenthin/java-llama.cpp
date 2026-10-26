/*
 * SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0
 */

/*
 * Augments GNU ld's default script for the s390x cross build (s390x-toolchain.cmake passes it with
 * -T; INSERT keeps the default script in force): every typeinfo name string (_ZTS*) lands 2-byte
 * aligned.
 *
 * The s390x ABI guarantees every global symbol an even address, and GCC relies on it: it loads a
 * symbol's address with LARL, whose R_390_PC32DBL relocation counts in halfwords. clang emits the
 * typeinfo name strings with the alignment of char instead (ItaniumCXXABI.cpp, GetAddrOfTypeName,
 * unchanged on LLVM main in 2026-10), and as COMDAT, so the copy the link keeps is the first one --
 * clang's, from our objects. gcc-toolset-14's libstdc++_nonshared.a, which the toolset's libstdc++.so
 * script adds to every link, is GCC code: fs_path80.o's _M_get_deleter loads
 * _ZTSSt19_Sp_make_shared_tag with LARL, and the link fails with "misaligned symbol ... for
 * relocation R_390_PC32DBL" whenever the sizes of the strings before it add up to an odd number. A
 * layout lottery: green on main, red on the next change (PR #577, the jllama_test link).
 *
 * SUBALIGN(2) sets the alignment of every input section of this output section to 2. Only typeinfo
 * name strings are matched; they are char arrays read byte-wise, and 2 is all the ABI promises
 * for them (GCC emits them with 2). INSERT BEFORE .rodata puts the statement ahead of the default
 * .rodata pattern (.rodata.*), so it is the first to match these sections. -fdata-sections (in the
 * toolchain) gives every typeinfo name such a section, those of classes with a key function and of
 * internal types included, and verify-native-deps.py fails a library with one at an odd address.
 */
SECTIONS
{
  .rodata.jllama_typeinfo_names : SUBALIGN(2) { *(.rodata._ZTS*) }
}
INSERT BEFORE .rodata;
