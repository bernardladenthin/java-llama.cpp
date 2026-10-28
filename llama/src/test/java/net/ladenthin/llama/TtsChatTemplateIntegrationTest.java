// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import net.ladenthin.llama.exception.LlamaException;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/**
 * Real-model coverage for a {@link TextToSpeech} pipeline whose GGUF carries a {@code "tts"} chat
 * template, LFM3-Audio (llama.cpp b11941, #31257): the engine speaks the prompt rendered from that
 * template with the text, the language and the voice, as llama.cpp's own {@code /tts} route and
 * {@code llama-tts} do. Before, the engine handed the raw text to the pipeline, which is not what
 * such a model expects, and had no way to pass a voice.
 *
 * <p>Not in CI's model set (1.4&nbsp;GB, and Qwen3-TTS already covers the pipeline itself): the test
 * self-skips unless {@link TestConstants#PROP_TTS_TEMPLATE_MODEL} and
 * {@link TestConstants#PROP_TTS_TEMPLATE_MMPROJ} name the pair, e.g. {@code LFM3-Audio-2B-Q4_0.gguf}
 * and {@code mmproj-LFM3-Audio-2B-Q8_0.gguf} of {@code LiquidAI/LFM3-Audio-2B-GGUF}.
 */
public class TtsChatTemplateIntegrationTest {

    private static final int WAV_HEADER_BYTES = 44;

    /** LFM3-Audio frames are 80 ms each: about two seconds, enough for a short sentence to start. */
    private static final int MAX_FRAMES = 25;

    private static TextToSpeech tts;

    @BeforeAll
    public static void setup() {
        String model = TestConstants.resolveModelProperty(TestConstants.PROP_TTS_TEMPLATE_MODEL);
        String mmproj = TestConstants.resolveModelProperty(TestConstants.PROP_TTS_TEMPLATE_MMPROJ);
        Assumptions.assumeTrue(
                model != null && !model.isEmpty(),
                "TTS model with a chat template not set (-D" + TestConstants.PROP_TTS_TEMPLATE_MODEL + "=...)");
        Assumptions.assumeTrue(
                mmproj != null && !mmproj.isEmpty(),
                "its mmproj not set (-D" + TestConstants.PROP_TTS_TEMPLATE_MMPROJ + "=...)");
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

    @Test
    @Timeout(value = 10, unit = TimeUnit.MINUTES)
    @DisplayName("the language code and voice reach the template: a well-formed, audible WAV")
    public void synthesizesWithLanguageAndVoice() {
        byte[] wav = tts.synthesize("Hello from Java.", null, "en", "male", MAX_FRAMES, 4, 1);
        assertWellFormedAudibleWav(wav);
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.MINUTES)
    @DisplayName("an empty language and voice leave the template's defaults")
    public void synthesizesWithTheTemplatesDefaults() {
        byte[] wav = tts.synthesize("Hello from Java.", null, null, null, MAX_FRAMES, 4, 1);
        assertWellFormedAudibleWav(wav);
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.MINUTES)
    @DisplayName("the voice changes the audio: female and male differ for the same text and seed")
    public void theVoiceChangesTheAudio() {
        byte[] female = tts.synthesize("Good morning.", null, "en", "female", MAX_FRAMES, 4, 7);
        byte[] male = tts.synthesize("Good morning.", null, "en", "male", MAX_FRAMES, 4, 7);
        assertWellFormedAudibleWav(female);
        assertWellFormedAudibleWav(male);
        assertFalse(
                Arrays.equals(pcm(female), pcm(male)),
                "the voice must reach the pipeline: two voices gave the same samples");
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.MINUTES)
    @DisplayName("the same text, voice and seed give the same audio")
    public void isReproducibleForTheSameSeed() {
        byte[] first = tts.synthesize("Good morning.", null, "en", "female", MAX_FRAMES, 4, 3);
        byte[] second = tts.synthesize("Good morning.", null, "en", "female", MAX_FRAMES, 4, 3);
        assertArrayEquals(first, second, "same text, voice and seed must give the same WAV");
    }

    @Test
    @DisplayName("no voice cloning: the pair reports it and refuses a speaker reference")
    public void refusesASpeakerReference() {
        assertFalse(tts.supportsSpeakerReference(), "LFM3-Audio has no voice encoder");
        assertThrows(
                LlamaException.class,
                () -> tts.synthesize("Hello.", TestConstants.DEFAULT_AUDIO_INPUT_PATH, "en", null, MAX_FRAMES, 4, 1));
    }

    /** RIFF/WAVE, PCM mono 16-bit at a positive rate, consistent sizes and at least one non-zero sample. */
    private static void assertWellFormedAudibleWav(byte[] wav) {
        assertTrue(wav.length > WAV_HEADER_BYTES, "WAV must carry samples; got " + wav.length + " bytes");
        assertEquals("RIFF", new String(wav, 0, 4, StandardCharsets.US_ASCII), "RIFF magic");
        assertEquals("WAVE", new String(wav, 8, 4, StandardCharsets.US_ASCII), "WAVE magic");
        ByteBuffer header = ByteBuffer.wrap(wav).order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(1, header.getShort(20) & 0xFFFF, "audio format must be PCM (1)");
        assertEquals(1, header.getShort(22) & 0xFFFF, "must be mono");
        assertTrue(header.getInt(24) > 0, "sample rate must be positive");
        assertEquals(16, header.getShort(34) & 0xFFFF, "must be 16-bit samples");
        assertEquals(wav.length - WAV_HEADER_BYTES, header.getInt(40), "data chunk size");
        boolean audible = false;
        for (int i = WAV_HEADER_BYTES; i < wav.length && !audible; i++) {
            audible = wav[i] != 0;
        }
        assertTrue(audible, "the PCM must not be pure silence");
    }

    private static byte[] pcm(byte[] wav) {
        return Arrays.copyOfRange(wav, WAV_HEADER_BYTES, wav.length);
    }
}
