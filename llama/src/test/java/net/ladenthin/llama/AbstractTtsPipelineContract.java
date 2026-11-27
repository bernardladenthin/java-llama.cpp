// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTimeout;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.TimeUnit;
import net.ladenthin.llama.exception.InvalidRequestException;
import net.ladenthin.llama.parameters.SpeechParameters;
import org.jspecify.annotations.Nullable;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.Timeout;

/**
 * What every speech pipeline of llama.cpp has to do through {@link TextToSpeech}, whichever it is:
 * Pocket TTS, Chatterbox (b12097), Drama-3 (b12116), Soprano. {@link TextToSpeech} drives
 * {@code mtmd_helper::gen_audio} the same way for every pipeline (only whether one clones a voice from a
 * clip is decided by its type), so the assertions are the ones that hold for all of them and read the
 * rest from the model: its sample rate from the WAV header, its
 * voices from {@link TextToSpeech#getVoices()}, voice cloning from
 * {@link TextToSpeech#supportsSpeakerReference()}.
 *
 * <p>One subclass per model pair: {@code PocketTtsIntegrationTest} and
 * {@code ChatterboxIntegrationTest} run CI's pairs, {@code TtsPipelineIntegrationTest} a pair named by
 * properties. Qwen3-TTS has {@code TtsIntegrationTest}, LFM3-Audio
 * {@code TtsChatTemplateIntegrationTest}.</p>
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public abstract class AbstractTtsPipelineContract {

    private static final int WAV_HEADER_BYTES = 44;
    private static final int MAX_FRAMES = 40;
    /** What one predefined voice's synthesis may take, a few times the slowest CI runner's. */
    private static final Duration PER_VOICE = Duration.ofMinutes(2);

    private TextToSpeech tts;

    /**
     * The backbone GGUF of the pipeline under test.
     *
     * @return its path, or {@code null} when none is configured
     */
    protected abstract String modelPath();

    /**
     * The mmproj GGUF of the pipeline under test.
     *
     * @return its path, or {@code null} when none is configured
     */
    protected abstract String mmprojPath();

    /**
     * Whether the pair under test clones a voice from a reference clip, as far as the test knows it:
     * CI's pairs say so, a pair named by properties does not.
     *
     * @return the expected {@link TextToSpeech#supportsSpeakerReference()}, or {@code null} when unknown
     */
    protected @Nullable Boolean clonesVoice() {
        return null;
    }

    @BeforeAll
    public void setup() {
        String model = modelPath();
        String mmproj = mmprojPath();
        Assumptions.assumeTrue(model != null && new File(model).exists(), "speech model not set or missing: " + model);
        Assumptions.assumeTrue(mmproj != null && new File(mmproj).exists(), "its mmproj not set or missing: " + mmproj);
        tts = new TextToSpeech(model, mmproj, Integer.getInteger(TestConstants.PROP_TEST_NGL, 0), 0);
    }

    /**
     * The pair under test, for the tests a subclass adds about what only its pipeline does.
     *
     * @return the loaded pair
     */
    protected TextToSpeech tts() {
        return tts;
    }

    @AfterAll
    public void tearDown() {
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
    @Timeout(value = 60, unit = TimeUnit.MINUTES)
    @DisplayName("predefined voices: every listed one speaks, an unknown one is refused with the list")
    public void everyListedVoiceSpeaks() {
        List<String> voices = tts.getVoices();
        Assumptions.assumeFalse(voices.isEmpty(), "this pipeline ships no predefined voices");
        // one synthesis per voice, each over that voice's own prompt rows, so the budget is per voice: CI's
        // ZONOS3 Mini lists 27 voices of 150 to 320 rows, about 34 s each on the macOS CPU runners (925 s
        // in all, past a 10-minute limit for the whole test on 2026-11-16 and 2026-11-23)
        for (String voice : voices) {
            assertWellFormedAudibleWav(assertTimeout(
                    PER_VOICE, () -> tts.synthesize("Hello.", null, null, voice, MAX_FRAMES, 4, 1), voice));
        }
        // the request's fault, so that a server answers 400: InvalidRequestException, not its parent
        InvalidRequestException unknown = assertThrows(
                InvalidRequestException.class, () -> tts.synthesize("Hello.", null, null, "nobody", MAX_FRAMES, 4, 1));
        assertTrue(unknown.getMessage().contains(voices.get(0)), unknown.getMessage());
    }

    @Test
    @DisplayName("the pair clones a voice exactly when its pipeline and projector can")
    public void theVoiceCloningCapabilityIsThePairs() {
        // the cloning case below takes whatever the pair reports, so a wrong report passes it
        Boolean expected = clonesVoice();
        Assumptions.assumeTrue(expected != null, "the pair under test is not one of CI's");
        assertEquals(expected, tts.supportsSpeakerReference());
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
                    InvalidRequestException.class,
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

    @Test
    @Timeout(value = 10, unit = TimeUnit.MINUTES)
    @DisplayName("top-p and a repetition penalty of the backbone sampler are accepted")
    public void topPAndARepetitionPenaltyAreAccepted() {
        assertWellFormedAudibleWav(tts.synthesize(
                "Hello.",
                SpeechParameters.builder()
                        .maxFrames(MAX_FRAMES)
                        .seed(1)
                        .topP(0.9f)
                        .repeatPenalty(1.1f)
                        .build()));
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.MINUTES)
    @DisplayName("a speed and loudness are spoken by a model with delivery control, else the request's fault")
    public void aDeliveryIsSpokenOrRefusedAsTheRequestsFault() {
        // b12431 (#32020): only Zonos3 controls its delivery; llama.cpp offers no capability to ask first
        SpeechParameters delivery = SpeechParameters.builder()
                .maxFrames(MAX_FRAMES)
                .seed(1)
                .speed(1.25f)
                .loudness(-20.0f)
                .build();
        try {
            // not "Hello.": ZONOS3 ends that within its codebook delay at a faster speed (measured: at 1.25
            // the 40 frames are silence, at 2.0 there is no audio at all), which says nothing about delivery
            assertWellFormedAudibleWav(tts.synthesize("Hello, this is a longer sentence to speak.", delivery));
        } catch (InvalidRequestException e) {
            assertTrue(e.getMessage().contains("does not support speed or loudness"), e.getMessage());
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.MINUTES)
    @DisplayName("a guidance scale and solver steps are spoken by a model that has them, else the request's fault")
    public void aGuidanceIsSpokenOrRefusedAsTheRequestsFault() {
        // b12459 (#32369): YuE2, MiniMax Music 3 and VoxCPM2 read them; llama.cpp offers no capability to ask
        for (SpeechParameters guidance : new SpeechParameters[] {
            SpeechParameters.builder()
                    .maxFrames(MAX_FRAMES)
                    .seed(1)
                    .cfgScale(2.0f)
                    .build(),
            SpeechParameters.builder()
                    .maxFrames(MAX_FRAMES)
                    .seed(1)
                    .flowSteps(4)
                    .build()
        }) {
            try {
                assertWellFormedAudibleWav(tts.synthesize("Hello, this is a longer sentence to speak.", guidance));
            } catch (InvalidRequestException e) {
                assertTrue(e.getMessage().contains("does not support cfg_scale or flow_steps"), e.getMessage());
            }
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.MINUTES)
    @DisplayName("options no model can take are the request's fault, and the model answers the next one")
    public void optionsOutOfRangeAreRefusedAsTheRequestsFault() {
        SpeechParameters[] refused = {
            SpeechParameters.builder().speed(0.0f).build(),
            SpeechParameters.builder().speed(Float.NaN).build(),
            SpeechParameters.builder().loudness(1.0f).build(),
            SpeechParameters.builder().topP(1.5f).build(),
            SpeechParameters.builder().repeatPenalty(0.0f).build(),
            SpeechParameters.builder().temperature(Float.NaN).build()
        };
        for (SpeechParameters parameters : refused) {
            assertThrows(
                    InvalidRequestException.class, () -> tts.synthesize("Hello.", parameters), parameters::toString);
        }
        assertWellFormedAudibleWav(tts.synthesize("Hello.", MAX_FRAMES, 4, 1));
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
