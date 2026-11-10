// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.concurrent.TimeUnit;
import net.ladenthin.llama.exception.LlamaException;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/**
 * What every speech pipeline of llama.cpp has to do through {@link TextToSpeech}, whichever it is:
 * Pocket TTS, Chatterbox (b12097), Drama-3 (b12116), Soprano. {@link TextToSpeech} drives
 * {@code mtmd_helper::gen_audio} without knowing the pipeline, so the assertions are the ones that
 * hold for all of them and read the rest from the model: its sample rate from the WAV header, its
 * voices from {@link TextToSpeech#getVoices()}, voice cloning from
 * {@link TextToSpeech#supportsSpeakerReference()}.
 *
 * <p>Not in CI's model set: self-skips unless {@link TestConstants#PROP_TTS_PIPELINE_MODEL} and
 * {@link TestConstants#PROP_TTS_PIPELINE_MMPROJ} name a pair. Qwen3-TTS is CI's
 * ({@code TtsIntegrationTest}), LFM3-Audio has {@code TtsChatTemplateIntegrationTest}.</p>
 */
public class TtsPipelineIntegrationTest {

    private static final int WAV_HEADER_BYTES = 44;
    private static final int MAX_FRAMES = 40;

    private static TextToSpeech tts;

    @BeforeAll
    public static void setup() {
        String model = TestConstants.resolveModelProperty(TestConstants.PROP_TTS_PIPELINE_MODEL);
        String mmproj = TestConstants.resolveModelProperty(TestConstants.PROP_TTS_PIPELINE_MMPROJ);
        Assumptions.assumeTrue(
                model != null && new File(model).exists(),
                "speech model not set or missing (-D" + TestConstants.PROP_TTS_PIPELINE_MODEL + "=...)");
        Assumptions.assumeTrue(
                mmproj != null && new File(mmproj).exists(),
                "its mmproj not set or missing (-D" + TestConstants.PROP_TTS_PIPELINE_MMPROJ + "=...)");
        tts = new TextToSpeech(model, mmproj, Integer.getInteger(TestConstants.PROP_TEST_NGL, 0), 0);
    }

    @AfterAll
    public static void tearDown() {
        if (tts != null) {
            tts.close();
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.MINUTES)
    @DisplayName("a synthesis is a well-formed, audible WAV at the pipeline's own sample rate")
    public void synthesizesAWellFormedAudibleWav() {
        assertWellFormedAudibleWav(tts.synthesize("Hello from Java.", MAX_FRAMES, 4, 1));
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.MINUTES)
    @DisplayName("a second synthesis on the same instance works, and the same seed gives the same audio")
    public void synthesizesAgainAndReproducibly() {
        byte[] first = tts.synthesize("Good morning.", MAX_FRAMES, 4, 3);
        byte[] second = tts.synthesize("Good morning.", MAX_FRAMES, 4, 3);
        assertWellFormedAudibleWav(first);
        assertArrayEquals(first, second, "same text and seed must give the same WAV");
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.MINUTES)
    @DisplayName("predefined voices: every listed one speaks, an unknown one is refused with the list")
    public void everyListedVoiceSpeaks() {
        List<String> voices = tts.getVoices();
        Assumptions.assumeFalse(voices.isEmpty(), "this pipeline ships no predefined voices");
        for (String voice : voices) {
            assertWellFormedAudibleWav(tts.synthesize("Hello.", null, null, voice, MAX_FRAMES, 4, 1));
        }
        LlamaException unknown = assertThrows(
                LlamaException.class, () -> tts.synthesize("Hello.", null, null, "nobody", MAX_FRAMES, 4, 1));
        assertTrue(unknown.getMessage().contains(voices.get(0)), unknown.getMessage());
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.MINUTES)
    @DisplayName("a speaker reference is cloned or refused, as the pipeline says it can")
    public void aSpeakerReferenceIsTakenOrRefusedAsThePipelineReports() {
        if (tts.supportsSpeakerReference()) {
            assertWellFormedAudibleWav(
                    tts.synthesize("Hello.", TestConstants.DEFAULT_AUDIO_INPUT_PATH, null, MAX_FRAMES, 4, 1));
        } else {
            assertThrows(
                    LlamaException.class,
                    () -> tts.synthesize("Hello.", TestConstants.DEFAULT_AUDIO_INPUT_PATH, null, MAX_FRAMES, 4, 1));
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.MINUTES)
    @DisplayName("the audio frame temperature is accepted: greedy, and the model's default")
    public void aTemperatureIsAccepted() {
        // only a pipeline that samples its frames reads it (Pocket TTS); the others must not fail on it
        assertWellFormedAudibleWav(tts.synthesize("Hello.", null, null, null, MAX_FRAMES, 4, 1, 0.0f));
        assertWellFormedAudibleWav(
                tts.synthesize("Hello.", null, null, null, MAX_FRAMES, 4, 1, TextToSpeech.MODEL_DEFAULT_TEMPERATURE));
    }

    /** RIFF/WAVE, PCM mono 16-bit at a positive rate, consistent sizes and at least one non-zero sample. */
    private static void assertWellFormedAudibleWav(byte[] wav) {
        assertTrue(wav.length > WAV_HEADER_BYTES, "WAV must carry samples; got " + wav.length + " bytes");
        assertEquals("RIFF", new String(wav, 0, 4, StandardCharsets.US_ASCII), "RIFF magic");
        assertEquals("WAVE", new String(wav, 8, 4, StandardCharsets.US_ASCII), "WAVE magic");
        ByteBuffer header = ByteBuffer.wrap(wav).order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(1, header.getShort(20) & 0xFFFF, "audio format must be PCM (1)");
        assertEquals(1, header.getShort(22) & 0xFFFF, "must be mono");
        int rate = header.getInt(24);
        assertTrue(rate >= 8000 && rate <= 96000, "an audio sample rate, got " + rate);
        assertEquals(rate * 2, header.getInt(28), "byte rate of 16-bit mono");
        assertEquals(16, header.getShort(34) & 0xFFFF, "must be 16-bit samples");
        assertEquals(wav.length - WAV_HEADER_BYTES, header.getInt(40), "data chunk size");
        boolean audible = false;
        for (int i = WAV_HEADER_BYTES; i < wav.length && !audible; i++) {
            audible = wav[i] != 0;
        }
        assertTrue(audible, "the PCM must not be pure silence");
    }
}
