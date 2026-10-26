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
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/**
 * Real-model coverage for {@link TextToSpeech} (Qwen3-TTS audio output, llama.cpp's upstream
 * {@code mtmd_helper::gen_audio} pipeline). Loads the backbone+mmproj TTS pipeline once and
 * synthesizes short clips, with the model's default voice and with a voice cloned from the committed
 * {@link TestConstants#DEFAULT_AUDIO_INPUT_PATH} clip, checking the WAV container is well-formed.
 *
 * <p>Uses CI's Qwen3-TTS backbone + mmproj ({@link TestConstants#DEFAULT_TTS_MODEL_PATH},
 * {@link TestConstants#DEFAULT_TTS_MMPROJ_PATH}) unless {@link TestConstants#PROP_TTS_MODEL} /
 * {@link TestConstants#PROP_TTS_MMPROJ} name others, and self-skips when a file is missing.
 */
public class TtsIntegrationTest {

    /** Canonical RIFF/WAVE header size in bytes (16-bit PCM, no extra chunks). */
    private static final int WAV_HEADER_BYTES = 44;

    /**
     * Frame cap of the cloned-voice synthesis. Qwen3-TTS runs at 12 frames per second, so this is
     * about four seconds of audio at most: enough to prove the reference clip was encoded and used,
     * short enough to keep the CPU-only CI job quick.
     */
    private static final int CLONED_MAX_FRAMES = 48;

    private static TextToSpeech tts;

    @BeforeAll
    public static void setup() {
        String model =
                TestConstants.resolveModelProperty(TestConstants.PROP_TTS_MODEL, TestConstants.DEFAULT_TTS_MODEL_PATH);
        String mmproj = TestConstants.resolveModelProperty(
                TestConstants.PROP_TTS_MMPROJ, TestConstants.DEFAULT_TTS_MMPROJ_PATH);
        Assumptions.assumeTrue(
                model != null && !model.isEmpty(), "TTS model not set (-D" + TestConstants.PROP_TTS_MODEL + "=...)");
        Assumptions.assumeTrue(
                mmproj != null && !mmproj.isEmpty(),
                "TTS mmproj not set (-D" + TestConstants.PROP_TTS_MMPROJ + "=...)");
        Assumptions.assumeTrue(new File(model).exists(), "TTS model file missing: " + model);
        Assumptions.assumeTrue(new File(mmproj).exists(), "TTS mmproj file missing: " + mmproj);

        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, 0);
        tts = new TextToSpeech(model, mmproj, gpuLayers, 0);
    }

    @AfterAll
    public static void tearDown() {
        if (tts != null) {
            tts.close();
        }
    }

    private static TextToSpeech tts() {
        assertNotNull(tts, "the TTS pipeline must be loaded by setup()");
        return tts;
    }

    @Test
    @DisplayName("synthesize() returns a well-formed, non-silent 24 kHz mono 16-bit WAV")
    @Timeout(value = 300_000, unit = TimeUnit.MILLISECONDS)
    public void synthesizesWellFormedWav() {
        assertWellFormedWav(tts().synthesize("hello from llama"));
    }

    @Test
    @DisplayName("Qwen3-TTS reports that it clones a voice from a speaker reference")
    public void supportsSpeakerReference() {
        // llama.cpp's mtmd_helper_cap::audio_gen_spk_ref (b11901) is true for the Qwen3-TTS pipeline,
        // which encodes the reference clip with the speaker encoder bundled in its mmproj.
        assertTrue(tts().supportsSpeakerReference(), "Qwen3-TTS must accept a speaker reference");
    }

    @Test
    @DisplayName("synthesize() with a speaker reference clip returns a well-formed, non-silent WAV")
    @Timeout(value = 300_000, unit = TimeUnit.MILLISECONDS)
    public void synthesizesWithAClonedVoice() {
        String reference = TestConstants.resolveModelProperty(
                TestConstants.PROP_AUDIO_PATH, TestConstants.DEFAULT_AUDIO_INPUT_PATH);
        Assumptions.assumeTrue(new File(reference).exists(), "speaker reference clip missing: " + reference);

        // The committed clip is 2.6 s of 48 kHz stereo; llama.cpp decodes it at the projector's rate.
        assertWellFormedWav(tts().synthesize("hello from llama", reference, "english", CLONED_MAX_FRAMES, 4, 0));
    }

    private static void assertWellFormedWav(byte[] wav) {
        assertNotNull(wav, "WAV bytes must not be null");
        // A bare 44-byte header with no payload is not a valid clip: require real samples beyond it.
        assertTrue(wav.length > WAV_HEADER_BYTES, "WAV must carry a header plus samples; got " + wav.length + " bytes");

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
