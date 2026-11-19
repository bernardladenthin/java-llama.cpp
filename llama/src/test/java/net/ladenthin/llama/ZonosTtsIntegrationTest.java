// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.concurrent.TimeUnit;
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

    @Override
    protected Boolean clonesVoice() {
        // its projector carries the speaker encoder (clip.has_audio_encoder, zonos3_spkenc)
        return true;
    }
}
