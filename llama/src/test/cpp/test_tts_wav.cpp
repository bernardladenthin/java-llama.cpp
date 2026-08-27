// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
//
// Unit tests for the in-memory WAV writer (src/main/cpp/tts_wav.hpp) — our own code, not upstream.
// The Qwen3-TTS pipeline it pairs with (mtmd_helper::gen_audio) is entirely upstream-owned (no
// project-side DSP to unit-test here) and exercised end-to-end by the Java TtsIntegrationTest.

#include "tts_wav.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

using namespace jllama_tts;

namespace {
uint32_t read_u32(const std::vector<uint8_t> &b, size_t off) {
    return (uint32_t)b[off] | ((uint32_t)b[off + 1] << 8) | ((uint32_t)b[off + 2] << 16) | ((uint32_t)b[off + 3] << 24);
}
std::string read_tag(const std::vector<uint8_t> &b, size_t off) {
    return std::string(b.begin() + off, b.begin() + off + 4);
}

uint32_t read_u16(const std::vector<uint8_t> &b, size_t off) { return (uint32_t)b[off] | ((uint32_t)b[off + 1] << 8); }
} // namespace

TEST(TtsWav, HeaderAndPayloadAreWellFormed) {
    std::vector<float> pcm = {0.0f, 0.5f, -0.5f, 1.0f, -1.0f};
    std::vector<uint8_t> wav = pcm_to_wav16_bytes(pcm, 24000);

    // 44-byte header + 2 bytes per 16-bit sample.
    ASSERT_EQ(wav.size(), 44u + pcm.size() * 2);
    EXPECT_EQ(read_tag(wav, 0), "RIFF");
    EXPECT_EQ(read_tag(wav, 8), "WAVE");
    EXPECT_EQ(read_tag(wav, 12), "fmt ");
    EXPECT_EQ(read_tag(wav, 36), "data");
    EXPECT_EQ(read_u32(wav, 16), 16u);                             // PCM fmt-chunk size
    EXPECT_EQ(read_u32(wav, 24), 24000u);                          // sample rate
    EXPECT_EQ(read_u32(wav, 40), (uint32_t)(pcm.size() * 2));      // data size
    EXPECT_EQ(read_u32(wav, 4), 36u + (uint32_t)(pcm.size() * 2)); // RIFF chunk size

    // fmt-chunk fields written via put_u16 (offsets 20/22/32/34). No other test reads a u16 from the
    // header, so without these the s390x big-endian ctest gate cannot observe a put_u16 byte-order
    // regression at all -- every other assertion here is a u32 or an ASCII tag.
    EXPECT_EQ(read_u16(wav, 20), 1u);  // audio format = PCM
    EXPECT_EQ(read_u16(wav, 22), 1u);  // mono
    EXPECT_EQ(read_u16(wav, 32), 2u);  // block_align = num_channels * bits_per_sample/8
    EXPECT_EQ(read_u16(wav, 34), 16u); // bits_per_sample

    // byte_rate is the one computed u32 in the writer. Pin it at two sample rates so a hardcoded
    // constant cannot satisfy it and a dropped factor is caught.
    EXPECT_EQ(read_u32(wav, 28), 24000u * 2); // sample_rate * num_channels * bits_per_sample/8
    const std::vector<uint8_t> wav16k = pcm_to_wav16_bytes(pcm, 16000);
    EXPECT_EQ(read_u32(wav16k, 24), 16000u);
    EXPECT_EQ(read_u32(wav16k, 28), 16000u * 2);
}

TEST(TtsWav, ClampsAndEncodesSamplesLittleEndian) {
    std::vector<uint8_t> wav = pcm_to_wav16_bytes({0.0f, 1.0f, -1.0f, -2.0f}, 24000);
    // 0 -> 0; 1.0 -> 32767; -1.0 -> -32767 (= -1.0*32767, floor clamp not reached); -2.0 clamps to -32768.
    auto sample = [&](int i) -> int16_t {
        size_t off = 44 + i * 2;
        return (int16_t)((uint16_t)wav[off] | ((uint16_t)wav[off + 1] << 8));
    };
    EXPECT_EQ(sample(0), 0);
    EXPECT_EQ(sample(1), 32767);
    EXPECT_EQ(sample(2), -32767);
    EXPECT_EQ(sample(3), -32768);
}
