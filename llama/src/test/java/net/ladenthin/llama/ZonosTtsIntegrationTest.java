// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import net.ladenthin.llama.exception.InvalidRequestException;
import net.ladenthin.llama.parameters.SpeechParameters;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/**
 * The speech pipeline contract for ZONOS3 Mini, CI's pair ({@link TestConstants#DEFAULT_ZONOS_MODEL_PATH}),
 * and what only this pipeline does (llama.cpp b12431, #32020): it controls its delivery. {@code speed}
 * and {@code loudness} pick condition buckets of the checkpoint, so a slower speed speaks the same
 * text for longer and a loudness changes the audio -- the end of the generation is the model's
 * end-of-audio code, which the frame cap below leaves room for.
 */
public class ZonosTtsIntegrationTest extends AbstractTtsPipelineContract {

    @Override
    protected String modelPath() {
        return TestConstants.resolveModelProperty(
                TestConstants.PROP_ZONOS_MODEL, TestConstants.DEFAULT_ZONOS_MODEL_PATH);
    }

    @Override
    protected String mmprojPath() {
        return TestConstants.resolveModelProperty(
                TestConstants.PROP_ZONOS_MMPROJ, TestConstants.DEFAULT_ZONOS_MMPROJ_PATH);
    }

    /** 67 ms per frame: room for the sentence below at half speed, which ends on its own before. */
    private static final int MAX_FRAMES = 600;

    private static final String TEXT = "The old lighthouse keeper climbed the spiral stairs every evening.";

    private byte[] speak(float speed, float loudness) {
        return tts().synthesize(
                        TEXT,
                        SpeechParameters.builder()
                                .maxFrames(MAX_FRAMES)
                                .seed(7)
                                .speed(speed)
                                .loudness(loudness)
                                .build());
    }

    @Test
    @Timeout(value = 20, unit = TimeUnit.MINUTES)
    @DisplayName("a slower speed speaks the same text for longer than a faster one")
    public void aSlowerSpeedTakesLonger() {
        byte[] slow = speak(0.5f, SpeechParameters.MODEL_DEFAULT_LOUDNESS);
        byte[] fast = speak(2.0f, SpeechParameters.MODEL_DEFAULT_LOUDNESS);
        assertTrue(slow.length > fast.length, "slow " + slow.length + " bytes, fast " + fast.length + " bytes");
    }

    @Test
    @Timeout(value = 20, unit = TimeUnit.MINUTES)
    @DisplayName("a loudness changes the audio, and the same request gives the same audio again")
    public void aLoudnessChangesTheAudio() {
        byte[] modelsOwn = speak(SpeechParameters.MODEL_DEFAULT_SPEED, SpeechParameters.MODEL_DEFAULT_LOUDNESS);
        byte[] quiet = speak(SpeechParameters.MODEL_DEFAULT_SPEED, -30.0f);
        assertFalse(Arrays.equals(modelsOwn, quiet), "a loudness of -30 LUFS must change the audio");
        assertTrue(Arrays.equals(quiet, speak(SpeechParameters.MODEL_DEFAULT_SPEED, -30.0f)), "reproducible");
    }

    /**
     * ZONOS3 has no classifier-free guidance and no flow-matching solver (llama.cpp b12459 reads
     * {@code cfg_scale} and {@code flow_steps} for YuE2, MiniMax Music 3 and VoxCPM2 only), so
     * {@code set_input} refuses either: the request's fault, named in the message, and the same
     * request without them still speaks as before. The values travel through
     * {@link TextToSpeech#synthesize(String, SpeechParameters)} and JNI into {@code jllama_tts_options};
     * a field lost on the way would make the request succeed.
     */
    @Test
    @Timeout(value = 20, unit = TimeUnit.MINUTES)
    @DisplayName("a guidance scale or solver steps are refused by name: ZONOS3 has neither")
    public void aGuidanceScaleOrSolverStepsAreRefusedByName() {
        // 60 frames are enough to speak the start of the sentence; the refusal comes before the first
        SpeechParameters plain =
                SpeechParameters.builder().maxFrames(60).seed(7).build();
        byte[] before = tts().synthesize(TEXT, plain);

        InvalidRequestException cfgScale = assertThrows(
                InvalidRequestException.class,
                () -> tts().synthesize(TEXT, plain.toBuilder().cfgScale(2.0f).build()));
        InvalidRequestException flowSteps = assertThrows(
                InvalidRequestException.class,
                () -> tts().synthesize(TEXT, plain.toBuilder().flowSteps(4).build()));

        assertTrue(cfgScale.getMessage().contains("cfg_scale"), cfgScale.getMessage());
        assertTrue(flowSteps.getMessage().contains("flow_steps"), flowSteps.getMessage());
        assertArrayEquals(before, tts().synthesize(TEXT, plain), "the same request without them is unchanged");
    }

    @Override
    protected Boolean clonesVoice() {
        // its projector carries the speaker encoder (clip.has_audio_encoder, zonos3_spkenc)
        return true;
    }
}
