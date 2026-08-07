// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/**
 * Real-model coverage for {@link TextToSpeech} (Qwen3-TTS audio output, llama.cpp's upstream
 * {@code mtmd_helper::gen_audio} pipeline). Loads the backbone+mmproj TTS pipeline and synthesizes
 * a short clip, checking the WAV container is well-formed.
 *
 * <p>Self-skips when {@link TestConstants#PROP_TTS_MODEL} or {@link TestConstants#PROP_TTS_MMPROJ}
 * is unset or its file is missing, so it runs only where the (large) Qwen3-TTS backbone + mmproj
 * GGUFs have been staged.
 */
public class TtsIntegrationTest {

    /** Canonical RIFF/WAVE header size in bytes (16-bit PCM, no extra chunks). */
    private static final int WAV_HEADER_BYTES = 44;

    @Test
    @DisplayName("synthesize() returns a well-formed, non-silent 24 kHz mono 16-bit WAV")
    @Timeout(value = 300_000, unit = TimeUnit.MILLISECONDS)
    public void synthesizesWellFormedWav() {
        String model = System.getProperty(TestConstants.PROP_TTS_MODEL);
        String mmproj = System.getProperty(TestConstants.PROP_TTS_MMPROJ);
        Assumptions.assumeTrue(
                model != null && !model.isEmpty(), "TTS model not set (-D" + TestConstants.PROP_TTS_MODEL + "=...)");
        Assumptions.assumeTrue(
                mmproj != null && !mmproj.isEmpty(),
                "TTS mmproj not set (-D" + TestConstants.PROP_TTS_MMPROJ + "=...)");
        Assumptions.assumeTrue(new File(model).exists(), "TTS model file missing: " + model);
        Assumptions.assumeTrue(new File(mmproj).exists(), "TTS mmproj file missing: " + mmproj);

        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, 0);
        try (TextToSpeech tts = new TextToSpeech(model, mmproj, gpuLayers, 0)) {
            byte[] wav = tts.synthesize("hello from llama");

            assertNotNull(wav, "WAV bytes must not be null");
            // A bare 44-byte header with no payload is not a valid clip: require real samples beyond it.
            assertTrue(
                    wav.length > WAV_HEADER_BYTES,
                    "WAV must carry a header plus samples; got " + wav.length + " bytes");

            // RIFF/WAVE container magic.
            assertEquals("RIFF", tag(wav, 0), "RIFF magic");
            assertEquals("WAVE", tag(wav, 8), "WAVE magic");
            assertEquals("fmt ", tag(wav, 12), "fmt subchunk tag");
            assertEquals("data", tag(wav, 36), "data subchunk tag");

            // fmt fields must match the documented output format: 24 kHz mono 16-bit PCM. A mis-loaded
            // model that still framed a header would not silently pass with the wrong rate/channels.
            ByteBuffer header = ByteBuffer.wrap(wav).order(ByteOrder.LITTLE_ENDIAN);
            assertEquals(1, header.getShort(20) & 0xFFFF, "audio format must be PCM (1)");
            assertEquals(1, header.getShort(22) & 0xFFFF, "must be mono (1 channel)");
            assertEquals(24_000, header.getInt(24), "sample rate must be 24 kHz");
            assertEquals(16, header.getShort(34) & 0xFFFF, "must be 16-bit samples");

            // Declared chunk sizes must be self-consistent with the actual byte-array length.
            assertEquals(wav.length - 8, header.getInt(4), "RIFF chunk size must equal fileLength - 8");
            int dataSize = header.getInt(40);
            assertEquals(wav.length - WAV_HEADER_BYTES, dataSize, "data chunk size must equal fileLength - 44");
            assertEquals(0, dataSize % 2, "16-bit PCM data size must be even");

            // The clip must contain real audio, not just the zeroed 0.25 s lead-in (or the all-silent
            // buffer a mis-configured model could still frame inside an otherwise valid header). The
            // original `length > 44` check passed on a single padding byte; scan the PCM payload instead.
            assertTrue(
                    hasNonZeroSample(wav, WAV_HEADER_BYTES),
                    "synthesized PCM must contain audible (non-zero) samples, not pure silence");
        }
    }

    /** Reads the 4-byte ASCII chunk tag at {@code offset}. */
    private static String tag(byte[] wav, int offset) {
        return new String(wav, offset, 4, StandardCharsets.US_ASCII);
    }

    /** Returns {@code true} if any byte of the PCM payload at or after {@code from} is non-zero. */
    private static boolean hasNonZeroSample(byte[] wav, int from) {
        for (int i = from; i < wav.length; i++) {
            if (wav[i] != 0) {
                return true;
            }
        }
        return false;
    }
}
