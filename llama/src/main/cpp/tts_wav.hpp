// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
//
// In-memory 16-bit PCM WAV writer for the TTS engine. The JNI layer hands the bytes straight to
// Java, so — unlike the upstream CLI's file-writing save_wav16 — this builds the RIFF/WAVE container
// into a byte vector and returns it. Header-only so the engine and the C++ unit tests can both use it.

#ifndef JLLAMA_TTS_WAV_HPP
#define JLLAMA_TTS_WAV_HPP

#include <cstdint>
#include <vector>

namespace jllama_tts {

// Encode float PCM samples (range ~[-1, 1]) as a 16-bit WAV byte stream (returned, not written to a
// file). Samples are scaled by 32767 and clamped to the int16 range. With more than one channel the
// samples are interleaved (left, right, left, ...), as llama.cpp's PCM output is since b12459, and
// the header says so; a channel count below 1 is mono.
inline std::vector<uint8_t> pcm_to_wav16_bytes(const std::vector<float> &data, int sample_rate, int channels = 1) {
    const uint16_t num_channels = (uint16_t)(channels > 0 ? channels : 1);
    const uint16_t bits_per_sample = 16;
    const uint32_t data_size = (uint32_t)(data.size() * (bits_per_sample / 8));
    const uint32_t byte_rate = (uint32_t)sample_rate * num_channels * (bits_per_sample / 8);
    const uint16_t block_align = num_channels * (bits_per_sample / 8);
    const uint32_t chunk_size = 36 + data_size;

    std::vector<uint8_t> out;
    out.reserve(44 + data_size);

    auto put_u32 = [&out](uint32_t v) {
        out.push_back((uint8_t)(v & 0xff));
        out.push_back((uint8_t)((v >> 8) & 0xff));
        out.push_back((uint8_t)((v >> 16) & 0xff));
        out.push_back((uint8_t)((v >> 24) & 0xff));
    };
    auto put_u16 = [&out](uint16_t v) {
        out.push_back((uint8_t)(v & 0xff));
        out.push_back((uint8_t)((v >> 8) & 0xff));
    };
    auto put_tag = [&out](const char *tag) {
        out.push_back((uint8_t)tag[0]);
        out.push_back((uint8_t)tag[1]);
        out.push_back((uint8_t)tag[2]);
        out.push_back((uint8_t)tag[3]);
    };

    put_tag("RIFF");
    put_u32(chunk_size);
    put_tag("WAVE");
    put_tag("fmt ");
    put_u32(16); // PCM fmt-chunk size
    put_u16(1);  // audio format = PCM
    put_u16(num_channels);
    put_u32((uint32_t)sample_rate);
    put_u32(byte_rate);
    put_u16(block_align);
    put_u16(bits_per_sample);
    put_tag("data");
    put_u32(data_size);

    for (const float sample : data) {
        double scaled = (double)sample * 32767.0;
        scaled = scaled < -32768.0 ? -32768.0 : (scaled > 32767.0 ? 32767.0 : scaled);
        int16_t pcm = (int16_t)scaled;
        out.push_back((uint8_t)(pcm & 0xff));
        out.push_back((uint8_t)((pcm >> 8) & 0xff));
    }

    return out;
}

} // namespace jllama_tts

#endif // JLLAMA_TTS_WAV_HPP
