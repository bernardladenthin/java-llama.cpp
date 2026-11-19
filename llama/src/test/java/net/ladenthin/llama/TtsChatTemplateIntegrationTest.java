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
import java.util.List;
import java.util.concurrent.TimeUnit;
import net.ladenthin.llama.exception.InvalidRequestException;
import net.ladenthin.llama.exception.LlamaException;
import net.ladenthin.llama.parameters.SpeechParameters;
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
 * <p>Runs with CI's LFM3-Audio pair ({@code .github/models.csv}, feature {@code realtime}:
 * {@code LFM3-Audio-2B-Q4_0.gguf} and {@code mmproj-LFM3-Audio-2B-Q8_0.gguf} of
 * {@code LiquidAI/LFM3-Audio-2B-GGUF}) unless {@link TestConstants#PROP_TTS_TEMPLATE_MODEL} and
 * {@link TestConstants#PROP_TTS_TEMPLATE_MMPROJ} name another, and self-skips when the files are
 * missing.
 */
public class TtsChatTemplateIntegrationTest {

    private static final int WAV_HEADER_BYTES = 44;

    /** LFM3-Audio frames are 80 ms each: about two seconds, enough for a short sentence to start. */
    private static final int MAX_FRAMES = 25;

    private static TextToSpeech tts;

    @BeforeAll
    public static void setup() {
        String model = TestConstants.resolveModelProperty(
                TestConstants.PROP_TTS_TEMPLATE_MODEL, TestConstants.DEFAULT_REALTIME_MODEL_PATH);
        String mmproj = TestConstants.resolveModelProperty(
                TestConstants.PROP_TTS_TEMPLATE_MMPROJ, TestConstants.DEFAULT_REALTIME_MMPROJ_PATH);
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
        // its projector has an audio encoder (for speech in), but its pipeline ignores a clip: the voice is
        // named in the prompt
        assertFalse(tts.supportsSpeakerReference(), "LFM3-Audio clones no voice");
        // the request's fault, so that a server answers 400
        assertThrows(
                InvalidRequestException.class,
                () -> tts.synthesize("Hello.", TestConstants.DEFAULT_AUDIO_INPUT_PATH, "en", null, MAX_FRAMES, 4, 1));
    }

    /**
     * Since llama.cpp b12061 a model lists the voices its projector ships, and one that has some
     * speaks with one of them: the named one, or the first when the request names none. LFM3-Audio's
     * {@code female} and {@code male} are such voices (and still named in its template).
     */
    @Test
    @Timeout(value = 10, unit = TimeUnit.MINUTES)
    @DisplayName("predefined voices: listed, the first one is the default, an unknown one is refused by name")
    public void thePredefinedVoicesAreListedAndTheFirstIsTheDefault() {
        List<String> voices = tts.getVoices();
        Assumptions.assumeFalse(voices.isEmpty(), "this model ships no predefined voices");
        assertTrue(voices.contains("female") && voices.contains("male"), "voices: " + voices);

        byte[] unnamed = tts.synthesize("Good morning.", null, "en", null, MAX_FRAMES, 4, 5);
        byte[] first = tts.synthesize("Good morning.", null, "en", voices.get(0), MAX_FRAMES, 4, 5);
        assertWellFormedAudibleWav(unnamed);
        assertArrayEquals(first, unnamed, "without a voice the model speaks with its first one, " + voices.get(0));

        LlamaException unknown = assertThrows(
                LlamaException.class, () -> tts.synthesize("Hello.", null, "en", "nobody", MAX_FRAMES, 4, 1));
        assertTrue(unknown.getMessage().contains(voices.get(0)), unknown.getMessage());
    }

    /** The greedy request the raw-prompt cases compare: the same seed and frame cap each time. */
    private static SpeechParameters greedy(boolean rawPrompt) {
        return SpeechParameters.builder()
                .maxFrames(MAX_FRAMES)
                .seed(7)
                .temperature(0.0f)
                .rawPrompt(rawPrompt)
                .build();
    }

    /**
     * With {@code rawPrompt} the text is the backbone's prompt as given (llama.cpp b12494): the prompt
     * this model's {@code "tts"} template renders for a text, passed raw, is the same synthesis as that
     * text with the template -- so the flag crosses JNI and the template is skipped, nothing else.
     */
    @Test
    @Timeout(value = 10, unit = TimeUnit.MINUTES)
    @DisplayName("a raw prompt is the backbone's prompt as given: the rendered template, raw, is the same audio")
    public void aRawPromptIsTheTemplatesPromptAsGiven() {
        // what LFM3-Audio's tokenizer.chat_template.tts renders for "Good morning." with its defaults
        String rendered = "<|startoftext|><|im_start|>system\nPerform TTS in American English using a female voice."
                + "<|im_end|>\n<|im_start|>user\nGood morning.<|im_end|>\n<|im_start|>assistant\n<|audio_start|>";

        byte[] templated = tts.synthesize("Good morning.", greedy(false));
        byte[] raw = tts.synthesize(rendered, greedy(true));

        assertWellFormedAudibleWav(templated);
        assertArrayEquals(templated, raw, "the rendered prompt given raw must be the templated synthesis");
    }

    /**
     * The bare text as a raw prompt does not end in {@code <|audio_start|>}, and the pipeline refuses
     * it in {@code set_input}: a {@link LlamaException} that is no {@link InvalidRequestException}, as
     * upstream's {@code /tts} answers a refused prompt with a server error (500), not 400. The same text
     * with the template still speaks, and the model answers the next request as before.
     */
    @Test
    @Timeout(value = 10, unit = TimeUnit.MINUTES)
    @DisplayName("the bare text as a raw prompt is refused as a server error, and the template still speaks it")
    public void theBareTextAsARawPromptIsRefusedAndTheTemplatedSynthesisIsUnchanged() {
        byte[] before = tts.synthesize("Good morning.", greedy(false));

        LlamaException refused =
                assertThrows(LlamaException.class, () -> tts.synthesize("Good morning.", greedy(true)));

        assertFalse(refused instanceof InvalidRequestException, "a refused prompt is no request error: " + refused);
        assertTrue(refused.getMessage().contains("failed to set TTS input"), refused.getMessage());
        assertWellFormedAudibleWav(before);
        assertArrayEquals(before, tts.synthesize("Good morning.", greedy(false)), "the next synthesis is unchanged");
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
