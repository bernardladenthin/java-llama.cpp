// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import org.jspecify.annotations.Nullable;

/**
 * What a speech request may set besides its text, voice and language: the fields of llama.cpp's
 * {@code POST /tts} that a {@link SpeechBackend} can apply. Every one is optional; {@code null}
 * leaves the engine's own default. Immutable.
 */
public final class SpeechOptions {

    private static final SpeechOptions DEFAULTS =
            new SpeechOptions(null, null, null, null, null, null, null, null, null, null);

    private final byte @Nullable [] speakerReference;
    private final @Nullable Integer maxFrames;
    private final @Nullable Integer topK;
    private final @Nullable Integer seed;
    private final @Nullable Float temperature;
    private final @Nullable Float topP;
    private final @Nullable Float repeatPenalty;
    private final @Nullable Float speed;
    private final @Nullable Float loudness;
    private final @Nullable String speakerReferenceText;

    /**
     * Create the options of one request.
     *
     * @param speakerReference an audio file (WAV, MP3, ...) whose voice is cloned, or {@code null}
     * @param maxFrames the cap on the generated audio frames, or {@code null}
     * @param topK the top-k of the sampler, or {@code null}
     * @param seed the seed of the sampler, or {@code null}
     * @param temperature the temperature the audio frames are sampled at, or {@code null}
     */
    public SpeechOptions(
            byte @Nullable [] speakerReference,
            @Nullable Integer maxFrames,
            @Nullable Integer topK,
            @Nullable Integer seed,
            @Nullable Float temperature) {
        this(speakerReference, maxFrames, topK, seed, temperature, null, null, null, null);
    }

    /**
     * Create the options of one request, with the sampling and delivery fields of llama.cpp b12431.
     *
     * @param speakerReference an audio file (WAV, MP3, ...) whose voice is cloned, or {@code null}
     * @param maxFrames the cap on the generated audio frames, or {@code null}
     * @param topK the top-k of the sampler, or {@code null}
     * @param seed the seed of the sampler, or {@code null}
     * @param temperature the temperature the audio frames are sampled at, or {@code null}
     * @param topP the top-p of the sampler and the code predictor, or {@code null}
     * @param repeatPenalty the repetition penalty of the sampler, or {@code null}
     * @param speed the speaking speed relative to the model's pace, or {@code null}
     * @param loudness the target loudness in LUFS, or {@code null}
     */
    public SpeechOptions(
            byte @Nullable [] speakerReference,
            @Nullable Integer maxFrames,
            @Nullable Integer topK,
            @Nullable Integer seed,
            @Nullable Float temperature,
            @Nullable Float topP,
            @Nullable Float repeatPenalty,
            @Nullable Float speed,
            @Nullable Float loudness) {
        this(speakerReference, maxFrames, topK, seed, temperature, topP, repeatPenalty, speed, loudness, null);
    }

    private SpeechOptions(
            byte @Nullable [] speakerReference,
            @Nullable Integer maxFrames,
            @Nullable Integer topK,
            @Nullable Integer seed,
            @Nullable Float temperature,
            @Nullable Float topP,
            @Nullable Float repeatPenalty,
            @Nullable Float speed,
            @Nullable Float loudness,
            @Nullable String speakerReferenceText) {
        this.speakerReference = speakerReference == null ? null : speakerReference.clone();
        this.maxFrames = maxFrames;
        this.topK = topK;
        this.seed = seed;
        this.temperature = temperature;
        this.topP = topP;
        this.repeatPenalty = repeatPenalty;
        this.speed = speed;
        this.loudness = loudness;
        this.speakerReferenceText = speakerReferenceText;
    }

    /**
     * These options with the transcript of the speaker reference (llama.cpp b12453's
     * {@code speaker_ref_text}), which a model that puts it into its prompt (Fish Audio S2) uses.
     *
     * @param text the transcript, or {@code null} for none
     * @return a copy of these options with the transcript
     */
    public SpeechOptions withSpeakerReferenceText(@Nullable String text) {
        return new SpeechOptions(
                speakerReference, maxFrames, topK, seed, temperature, topP, repeatPenalty, speed, loudness, text);
    }

    /**
     * The options of a request that sets none.
     *
     * @return options with every value {@code null}
     */
    public static SpeechOptions defaults() {
        return DEFAULTS;
    }

    /**
     * The audio file whose voice is cloned.
     *
     * @return a copy of the file's bytes, or {@code null} for the model's own voice
     */
    public byte @Nullable [] getSpeakerReference() {
        return speakerReference == null ? null : speakerReference.clone();
    }

    /**
     * The cap on the generated audio frames.
     *
     * @return the cap, or {@code null} for the engine's default
     */
    public @Nullable Integer getMaxFrames() {
        return maxFrames;
    }

    /**
     * The top-k of the sampler.
     *
     * @return the top-k, or {@code null} for the engine's default
     */
    public @Nullable Integer getTopK() {
        return topK;
    }

    /**
     * The seed of the sampler.
     *
     * @return the seed, or {@code null} for the engine's default
     */
    public @Nullable Integer getSeed() {
        return seed;
    }

    /**
     * The temperature the audio frames are sampled at.
     *
     * @return the temperature, or {@code null} for the model's default
     */
    public @Nullable Float getTemperature() {
        return temperature;
    }

    /**
     * The top-p of the sampler and the acoustic code predictor.
     *
     * @return the top-p, or {@code null} for none
     */
    public @Nullable Float getTopP() {
        return topP;
    }

    /**
     * The repetition penalty of the sampler.
     *
     * @return the penalty, or {@code null} for none
     */
    public @Nullable Float getRepeatPenalty() {
        return repeatPenalty;
    }

    /**
     * The speaking speed relative to the model's own pace (a model with delivery control).
     *
     * @return the speed, or {@code null} for the model's own
     */
    public @Nullable Float getSpeed() {
        return speed;
    }

    /**
     * The target loudness in LUFS (a model with delivery control).
     *
     * @return the loudness, or {@code null} for the model's own
     */
    public @Nullable Float getLoudness() {
        return loudness;
    }

    /**
     * The transcript of the speaker reference (a model that puts it into its prompt, Fish Audio S2).
     *
     * @return the transcript, or {@code null} for none
     */
    public @Nullable String getSpeakerReferenceText() {
        return speakerReferenceText;
    }

    @Override
    public String toString() {
        return "SpeechOptions{speakerReference="
                + (speakerReference == null ? "none" : speakerReference.length + " bytes")
                + ", maxFrames=" + maxFrames
                + ", topK=" + topK
                + ", seed=" + seed
                + ", temperature=" + temperature
                + ", topP=" + topP
                + ", repeatPenalty=" + repeatPenalty
                + ", speed=" + speed
                + ", loudness=" + loudness
                + ", speakerReferenceText="
                + (speakerReferenceText == null ? "none" : speakerReferenceText.length() + " chars")
                + '}';
    }
}
