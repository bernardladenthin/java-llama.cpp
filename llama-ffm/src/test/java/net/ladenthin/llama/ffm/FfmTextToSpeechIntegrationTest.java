// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
package net.ladenthin.llama.ffm;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.greaterThan;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * {@link FfmTextToSpeech} over a real Qwen3-TTS pair (CI's backbone and mmproj, {@code
 * -Dnet.ladenthin.llama.ffm.tts.model} / {@code .tts.mmproj} for others): the speaker-reference
 * capability and a synthesis with a voice cloned from the clip the core's tests commit. Self-skips
 * without the natives or the models, which CI does not allow.
 */
class FfmTextToSpeechIntegrationTest {

    /** Canonical RIFF/WAVE header size in bytes (16-bit PCM, no extra chunks). */
    private static final int WAV_HEADER_BYTES = 44;

    /** About four seconds at Qwen3-TTS's 12 frames per second: enough to prove the clip was used. */
    private static final int MAX_FRAMES = 48;

    private static FfmTextToSpeech tts;

    @BeforeAll
    static void load() {
        Path model = FfmTestSupport.assumeFile(FfmTestSupport.TTS_MODEL_PROPERTY, FfmTestSupport.DEFAULT_TTS_MODEL);
        Path mmproj =
                FfmTestSupport.assumeFile(FfmTestSupport.TTS_MMPROJ_PROPERTY, FfmTestSupport.DEFAULT_TTS_MMPROJ);
        tts = FfmTextToSpeech.load(model.toString(), mmproj.toString(), 0, 0);
    }

    @AfterAll
    static void free() {
        if (tts != null) {
            tts.close();
        }
    }

    @Test
    void qwen3TtsClonesAVoiceFromASpeakerReference() {
        // mtmd_helper_cap::audio_gen_spk_ref through jllama_tts_supports_speaker_reference's bool slot
        assertTrue(tts.supportsSpeakerReference());
    }

    @Test
    void synthesisWithASpeakerReferenceReturnsAWav() {
        Path reference = FfmTestSupport.resolve(FfmTestSupport.SPEAKER_REFERENCE);
        byte[] wav = tts.synthesize("hello from llama", reference.toString(), "english", MAX_FRAMES, 4, 0);

        assertThat(wav.length, greaterThan(WAV_HEADER_BYTES));
        assertEquals("RIFF", new String(wav, 0, 4, StandardCharsets.US_ASCII));
        assertEquals("WAVE", new String(wav, 8, 4, StandardCharsets.US_ASCII));
    }
}
