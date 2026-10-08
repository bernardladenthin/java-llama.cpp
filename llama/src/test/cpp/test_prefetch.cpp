// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
//
// Runnable guard for patches/0017 (the x86 prefetch helper in ggml/src/ggml-cpu/arch/x86/).
//
// The patch fixes two independent defects in the four _mm_prefetch calls of
// ggml_vec_dot_q4_0_q8_0's SSSE3-without-AVX branch. Only one of them is a build error:
//
//   * The TYPE half is caught by the compiler. MSVC declares the intrinsic as
//     `void _mm_prefetch(char const *, int)`, and clang's own casting macro sits behind
//     `#ifndef _MSC_VER`, so plain clang on a *-windows-msvc target rejects a typed pointer
//     outright. Nothing here can test that; a dropped patch simply fails to compile the
//     x64 and sse42 CPU variants.
//
//   * The ARITHMETIC half is silent, wrong on every platform, and is what this file guards.
//     `&x[ib] + sizeof(block_q4_0)` adds sizeof() *elements*: 324 bytes ahead instead of the
//     next block, because ggml-common.h pins sizeof(block_q4_0) == 18 by static_assert.
//     ggml_prefetch_at() casts to `const char *` first, which is what makes the offset
//     byte-wise -- plainly what the call sites intend ("the next block", "two blocks ahead").
//
// The case that makes this test worth having: writing the cast around the whole expression,
// `(const char *)(&x[ib] + sizeof(block_q4_0))`, silences the compiler and keeps the wrong
// address. That form passes the build and fails here.

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include "prefetch.h"

#include <cstddef>
#include <type_traits>

#include <gtest/gtest.h>

namespace {

// The byte distance ggml_prefetch_at() produced, measured from the same base pointer.
template <typename Block> std::ptrdiff_t offset_of(const Block *base, std::size_t bytes) {
    return ggml_prefetch_at(base, bytes) - reinterpret_cast<const char *>(base);
}

} // namespace

// The sizes the arithmetic depends on. ggml-common.h asserts them itself, but a reader of this
// file needs them in front of them: 18 and 34 are what make 324 and 1156 wrong.
TEST(PrefetchAt, BlockSizesAreTheOnesTheOffsetsAssume) {
    EXPECT_EQ(sizeof(block_q4_0), 18u);
    EXPECT_EQ(sizeof(block_q8_0), 34u);
}

// One block ahead -- the first pair of call sites.
TEST(PrefetchAt, OneBlockAheadIsOneBlockInBytes) {
    const block_q4_0 x[4] = {};
    const block_q8_0 y[4] = {};

    EXPECT_EQ(offset_of(&x[0], sizeof(block_q4_0)), static_cast<std::ptrdiff_t>(sizeof(block_q4_0)));
    EXPECT_EQ(offset_of(&y[0], sizeof(block_q8_0)), static_cast<std::ptrdiff_t>(sizeof(block_q8_0)));
}

// Two blocks ahead -- the second pair, which the loop issues for ib + 1.
TEST(PrefetchAt, TwoBlocksAheadIsTwoBlocksInBytes) {
    const block_q4_0 x[8] = {};
    const block_q8_0 y[8] = {};

    EXPECT_EQ(offset_of(&x[0], 2 * sizeof(block_q4_0)), static_cast<std::ptrdiff_t>(2 * sizeof(block_q4_0)));
    EXPECT_EQ(offset_of(&y[0], 2 * sizeof(block_q8_0)), static_cast<std::ptrdiff_t>(2 * sizeof(block_q8_0)));
}

// The negative half, and the reason this file is not redundant with the build: the expression
// the patch replaces computes a different address, and a cast placed around the whole thing
// keeps computing it. These are the numbers a reviewer should see spelled out.
TEST(PrefetchAt, TheUnpatchedExpressionIsFarOff) {
    const block_q4_0 x[64] = {};
    const block_q8_0 y[64] = {};

    const char *base_x = reinterpret_cast<const char *>(&x[0]);
    const char *base_y = reinterpret_cast<const char *>(&y[0]);

    // What `&x[0] + sizeof(block_q4_0)` evaluates to: sizeof() ELEMENTS, not bytes.
    EXPECT_EQ(reinterpret_cast<const char *>(&x[0] + sizeof(block_q4_0)) - base_x, 324);
    EXPECT_EQ(reinterpret_cast<const char *>(&y[0] + sizeof(block_q8_0)) - base_y, 1156);
    EXPECT_EQ(reinterpret_cast<const char *>(&x[0] + 2 * sizeof(block_q4_0)) - base_x, 648);

    // ... and that it is not what the helper gives, i.e. the two forms are not interchangeable.
    EXPECT_NE(reinterpret_cast<const char *>(&x[0] + sizeof(block_q4_0)) - base_x,
              offset_of(&x[0], sizeof(block_q4_0)));
}

// ggml_prefetch_at() takes `const void *`, so it is usable from any block type without a cast
// at the call site, and it always returns `const char *` -- the type MSVC's _mm_prefetch wants.
TEST(PrefetchAt, ReturnsCharPointerForAnyBlockType) {
    const block_q4_0 x[2] = {};
    static_assert(std::is_same<decltype(ggml_prefetch_at(&x[0], 0)), const char *>::value,
                  "ggml_prefetch_at must return const char *, the type MSVC's _mm_prefetch takes");
    EXPECT_EQ(ggml_prefetch_at(&x[0], 0), reinterpret_cast<const char *>(&x[0]));
}
